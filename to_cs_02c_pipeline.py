#!/usr/bin/env python3
"""
TO-CS-02c: Within-N2 Micro-Arousal Dissociation Test
=====================================================
Tests whether Omega and kappa dissociate WITHIN N2 sleep, enabling the product test.

Pre-registered: 2026-02-08
Parameters: LOCKED (inherited from TO-CS-02b + new N2-specific definitions)

This script reuses the same ANPHY-Sleep data already extracted for TO-CS-02b.
No new downloads needed.

Usage:
    python to_cs_02c_pipeline.py

Requirements: same as TO-CS-02b (mne, numpy, scipy, pyinform, pandas)
"""

import os
import sys
import json
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.signal import welch, butter, filtfilt, hilbert

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "./anphy_sleep_data"
RESULTS_DIR = "./results_02c"

# ============================================================
# LOCKED PARAMETERS (inherited from TO-CS-02b)
# ============================================================

LOCKED = {
    # Epochs
    "epoch_length_sec": 10,
    "epoch_reject_uv": 300,
    "min_epochs_n2_total": 50,
    "min_epochs_pre_arousal": 5,
    "min_epochs_stable_n2": 20,

    # Spectral entropy
    "se_nperseg": 2048,
    "se_noverlap_frac": 0.5,

    # Frequency bands
    "band_full": (0.5, 45.0),
    "band_delta_free": (4.0, 40.0),

    # Transfer entropy
    "te_k": 3,
    "te_bins": 6,

    # Bootstrap
    "n_bootstrap": 10000,
    "bootstrap_seed": 42,

    # Multiple comparison
    "alpha": 0.05,
    "n_comparisons": 4,
    "alpha_corrected": 0.05 / 4,

    # Minimum subjects
    "min_subjects": 5,

    # NEW: TO-CS-02c specific (locked)
    "stop_rule_within_n2_corr": 0.6,
    "spindle_band": (12.0, 15.0),
    "spindle_threshold_sd": 2.0,
    "spindle_min_duration_sec": 0.5,
    "kcomplex_band": (0.5, 4.0),
    "kcomplex_min_ptp_uv": 75.0,
    "kcomplex_duration_range_sec": (0.5, 1.5),
    "pre_arousal_next_stages": ["N1", "Wake"],
    "stable_n2_buffer": 2,  # need >= 2 N2 epochs on each side
}

# Scoring file stage mapping
STAGE_MAP = {
    'W': 'Wake', 'R': 'REM', 'N1': 'N1', 'N2': 'N2', 'N3': 'N3',
    '0': 'Wake', '1': 'N1', '2': 'N2', '3': 'N3', '4': 'N3', '5': 'REM',
    'WAKE': 'Wake', 'REM': 'REM', 'Wake': 'Wake',
}

EXCLUDE_PATTERNS = [
    "EOG", "EMG", "ECG", "EKG", "ZY", "SO1", "SO2",
    "RLEG", "LLEG", "ChEMG", "Status", "STI",
    "FT11", "FT12", "TP11", "TP12", "F11", "F12", "P11", "P12",
]

ROI_DEFINITIONS = {
    "F_L":  ["Fp1", "AF3", "AF7", "F3", "F5", "F7", "F1", "F9"],
    "F_R":  ["Fp2", "AF4", "AF8", "F4", "F6", "F8", "F2", "FZ", "F10", "FPZ", "AFZ"],
    "C_L":  ["C3", "C5", "C1", "FC3", "FC5", "FC1", "CP3", "CP1", "T3", "T9", "FT9", "FT7", "TP9", "TP7"],
    "C_R":  ["C4", "C6", "C2", "FC4", "FC6", "FC2", "CP4", "CP2", "CZ", "FCZ", "CPZ", "T4", "T10", "FT8", "FT10", "TP8", "TP10"],
    "P_L":  ["P3", "P5", "P7", "P1", "CP5", "PO3", "PO7", "T5", "P9"],
    "P_R":  ["P4", "P6", "P8", "P2", "CP6", "PO4", "PO8", "PZ", "POZ", "T6", "P10"],
    "O_L":  ["O1", "PO7", "P7"],
    "O_R":  ["O2", "PO8", "OZ", "P8"],
}


# ============================================================
# HELPERS (same as TO-CS-02b)
# ============================================================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def find_edf_files(data_dir):
    files = []
    for p in ["**/*.edf", "**/*.EDF"]:
        files.extend(glob.glob(os.path.join(data_dir, p), recursive=True))
    return sorted(set(files))


def is_eeg_channel(ch_name):
    for pattern in EXCLUDE_PATTERNS:
        if pattern.lower() in ch_name.lower():
            return False
    return True


def map_channels_to_rois(ch_names):
    def norm(name):
        return name.strip().replace('.', '').replace('-', '').replace(' ', '').upper()
    ch_norm = {norm(ch): i for i, ch in enumerate(ch_names)}
    roi_mapping = {}
    mapped_channels = {}
    for roi_name, roi_channels in ROI_DEFINITIONS.items():
        indices = []
        matched = []
        for rc in roi_channels:
            rc_n = norm(rc)
            if rc_n in ch_norm:
                indices.append(ch_norm[rc_n])
                matched.append(ch_names[ch_norm[rc_n]])
        if len(indices) >= 2:
            roi_mapping[roi_name] = indices
            mapped_channels[roi_name] = matched
    return roi_mapping, mapped_channels


def load_scoring(txt_path):
    scoring = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            for p in parts:
                p_clean = p.strip()
                if p_clean in STAGE_MAP:
                    scoring.append(STAGE_MAP[p_clean])
                    break
    return scoring


def load_and_preprocess(filepath, target_sfreq=250):
    import mne
    mne.set_log_level('WARNING')
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    except Exception as e:
        log(f"  ERROR loading: {e}")
        return None, None

    # Strip -Ref suffix
    rename = {ch: ch.replace('-Ref', '') for ch in raw.ch_names if ch.endswith('-Ref')}
    if rename:
        raw.rename_channels(rename)

    eeg_picks = [ch for ch in raw.ch_names if is_eeg_channel(ch)]
    raw.pick_channels(eeg_picks)

    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq)

    raw.filter(0.5, 45.0, verbose=False)
    return raw, raw.ch_names


# ============================================================
# SPECTRAL ENTROPY & TRANSFER ENTROPY (same as TO-CS-02b)
# ============================================================

def spectral_entropy(signal, sfreq, band):
    nperseg = min(LOCKED["se_nperseg"], len(signal))
    noverlap = int(nperseg * LOCKED["se_noverlap_frac"])
    freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    psd_band = psd[mask]
    if psd_band.sum() == 0:
        return 0.0
    psd_norm = psd_band / psd_band.sum()
    psd_norm = psd_norm[psd_norm > 0]
    H = -np.sum(psd_norm * np.log2(psd_norm))
    H_max = np.log2(len(psd_norm))
    return H / H_max if H_max > 0 else 0.0


def discretise(signal, n_bins):
    sig_min, sig_max = signal.min(), signal.max()
    if sig_max == sig_min:
        return np.zeros(len(signal), dtype=int)
    bins = np.linspace(sig_min, sig_max, n_bins + 1)
    return np.digitize(signal, bins[1:-1]).astype(int)


def transfer_entropy_pair(source, target, k, n_bins):
    try:
        from pyinform import transfer_entropy as te_func
        src_d = discretise(source, n_bins)
        tgt_d = discretise(target, n_bins)
        return te_func(src_d, tgt_d, k=k)
    except ImportError:
        return _manual_te(source, target, k, n_bins)
    except Exception:
        return 0.0


def _manual_te(source, target, k, n_bins):
    from collections import Counter
    src_d = discretise(source, n_bins)
    tgt_d = discretise(target, n_bins)
    n = len(src_d)
    if n <= k + 1:
        return 0.0
    joint_xyz = Counter()
    joint_yz = Counter()
    joint_yz_only = Counter()
    joint_y = Counter()
    for t in range(k, n - 1):
        yf = tgt_d[t + 1]
        yp = tuple(tgt_d[t - k + 1:t + 1])
        xp = tuple(src_d[t - k + 1:t + 1])
        joint_xyz[(yf, yp, xp)] += 1
        joint_yz[(yf, yp)] += 1
        joint_yz_only[yp] += 1
        joint_y[(yp, xp)] += 1
    total = sum(joint_xyz.values())
    if total == 0:
        return 0.0
    te = 0.0
    for (yf, yp, xp), count in joint_xyz.items():
        p_xyz = count / total
        p_yf_ypxp = count / joint_y[(yp, xp)] if joint_y[(yp, xp)] > 0 else 0
        p_yf_yp = joint_yz[(yf, yp)] / joint_yz_only[yp] if joint_yz_only[yp] > 0 else 0
        if p_yf_ypxp > 0 and p_yf_yp > 0:
            te += p_xyz * np.log2(p_yf_ypxp / p_yf_yp)
    return max(te, 0.0)


# ============================================================
# N2 EPOCH CLASSIFICATION
# ============================================================

def classify_n2_epochs(scoring, n2_epoch_indices):
    """
    Classify N2 epochs as pre-arousal, stable, or other.

    scoring: list of stage labels per 30s scoring epoch
    n2_epoch_indices: list of (scoring_epoch_index, sub_epoch_index) for each 10s epoch

    Returns: dict with 'pre_arousal' and 'stable' lists of epoch indices
    """
    n_scoring = len(scoring)
    buf = LOCKED["stable_n2_buffer"]
    next_stages = LOCKED["pre_arousal_next_stages"]

    pre_arousal = []
    stable = []
    other_n2 = []

    for ep_idx, (score_idx, sub_idx) in enumerate(n2_epoch_indices):
        # Pre-arousal: next scoring epoch is N1 or Wake
        # Only the LAST sub-epoch (sub_idx == 2) of a scoring epoch counts as "pre-arousal"
        # because it's the 10s segment closest to the transition
        is_pre_arousal = False
        if sub_idx == 2 and score_idx + 1 < n_scoring:
            next_stage = scoring[score_idx + 1]
            if next_stage in next_stages:
                is_pre_arousal = True

        # Stable: surrounded by buffer N2 epochs on each side
        is_stable = True
        for offset in range(-buf, buf + 1):
            check_idx = score_idx + offset
            if check_idx < 0 or check_idx >= n_scoring:
                is_stable = False
                break
            if scoring[check_idx] != "N2":
                is_stable = False
                break

        if is_pre_arousal:
            pre_arousal.append(ep_idx)
        elif is_stable:
            stable.append(ep_idx)
        else:
            other_n2.append(ep_idx)

    return {
        "pre_arousal": pre_arousal,
        "stable": stable,
        "other": other_n2,
    }


def detect_spindle_epochs(epochs_data, sfreq, roi_signals_all):
    """
    Detect epochs containing sleep spindles.
    Uses sigma band (12-15 Hz) power on central/parietal ROIs.

    Returns: list of epoch indices containing spindles
    """
    band = LOCKED["spindle_band"]
    thresh_sd = LOCKED["spindle_threshold_sd"]
    min_dur = LOCKED["spindle_min_duration_sec"]
    min_samples = int(min_dur * sfreq)

    # Use central ROIs for spindle detection (C_L, C_R, P_L, P_R)
    spindle_rois = [r for r in ["C_L", "C_R", "P_L", "P_R"] if r in roi_signals_all]
    if not spindle_rois:
        return []

    # Bandpass filter design
    nyq = sfreq / 2
    b, a = butter(4, [band[0]/nyq, band[1]/nyq], btype='band')

    # Compute sigma RMS for all epochs
    sigma_rms_all = []
    n_epochs = roi_signals_all[spindle_rois[0]].shape[0]

    for ep in range(n_epochs):
        rms_vals = []
        for roi in spindle_rois:
            sig = roi_signals_all[roi][ep]
            sig_filt = filtfilt(b, a, sig)
            rms = np.sqrt(np.mean(sig_filt**2))
            rms_vals.append(rms)
        sigma_rms_all.append(np.mean(rms_vals))

    sigma_rms_all = np.array(sigma_rms_all)
    mean_rms = np.mean(sigma_rms_all)
    sd_rms = np.std(sigma_rms_all)

    # Check which epochs exceed threshold AND have sustained spindle
    spindle_epochs = []
    for ep in range(n_epochs):
        has_spindle = False
        for roi in spindle_rois:
            sig = roi_signals_all[roi][ep]
            sig_filt = filtfilt(b, a, sig)
            envelope = np.abs(hilbert(sig_filt))
            above = envelope > (mean_rms + thresh_sd * sd_rms) * np.sqrt(2)  # RMS to amplitude

            # Check for sustained run
            run_length = 0
            max_run = 0
            for val in above:
                if val:
                    run_length += 1
                    max_run = max(max_run, run_length)
                else:
                    run_length = 0

            if max_run >= min_samples:
                has_spindle = True
                break

        if has_spindle:
            spindle_epochs.append(ep)

    return spindle_epochs


def detect_kcomplex_epochs(epochs_data, sfreq, roi_mapping, raw_epochs):
    """
    Detect epochs containing K-complexes.
    Uses delta band (0.5-4 Hz) in frontal ROIs.

    Returns: list of epoch indices containing K-complexes
    """
    band = LOCKED["kcomplex_band"]
    min_ptp = LOCKED["kcomplex_min_ptp_uv"] * 1e-6  # convert to V
    dur_range = LOCKED["kcomplex_duration_range_sec"]

    frontal_rois = [r for r in ["F_L", "F_R"] if r in roi_mapping]
    if not frontal_rois:
        return []

    nyq = sfreq / 2
    b, a = butter(4, [band[0]/nyq, band[1]/nyq], btype='band')

    kc_epochs = []
    n_epochs = raw_epochs.shape[0]

    for ep in range(n_epochs):
        has_kc = False
        for roi in frontal_rois:
            ch_indices = roi_mapping[roi]
            roi_signal = np.mean(raw_epochs[ep, ch_indices, :], axis=0)
            sig_filt = filtfilt(b, a, roi_signal)

            # Look for sharp negative-then-positive deflection
            ptp = np.ptp(sig_filt)
            if ptp >= min_ptp:
                neg_idx = np.argmin(sig_filt)
                pos_idx = np.argmax(sig_filt)

                if pos_idx > neg_idx:  # neg then pos (classic KC shape)
                    duration = (pos_idx - neg_idx) / sfreq
                    if dur_range[0] <= duration <= dur_range[1]:
                        has_kc = True
                        break

        if has_kc:
            kc_epochs.append(ep)

    return kc_epochs


# ============================================================
# COMPUTE MEASURES PER EPOCH
# ============================================================

def compute_epoch_measures(roi_signals, sfreq):
    """
    Compute Omega and kappa for each epoch individually.
    Returns arrays of per-epoch values.
    """
    roi_names = list(roi_signals.keys())
    n_rois = len(roi_names)
    n_epochs = roi_signals[roi_names[0]].shape[0]

    omega_full_epochs = np.zeros(n_epochs)
    omega_df_epochs = np.zeros(n_epochs)
    kappa_epochs = np.zeros(n_epochs)
    te_p2c_epochs = np.zeros(n_epochs)
    te_c2p_epochs = np.zeros(n_epochs)

    log(f"    Computing per-epoch Omega and kappa for {n_epochs} epochs...")

    for ep in range(n_epochs):
        if ep % 50 == 0 and ep > 0:
            log(f"      epoch {ep}/{n_epochs}...")

        # Omega per epoch
        se_full = []
        se_df = []
        for roi in roi_names:
            se_full.append(spectral_entropy(roi_signals[roi][ep], sfreq, LOCKED["band_full"]))
            se_df.append(spectral_entropy(roi_signals[roi][ep], sfreq, LOCKED["band_delta_free"]))
        omega_full_epochs[ep] = np.mean(se_full)
        omega_df_epochs[ep] = np.mean(se_df)

        # kappa per epoch (mean TE across all pairs)
        te_vals = []
        te_p2c_vals = []
        te_c2p_vals = []

        for i, src in enumerate(roi_names):
            for j, tgt in enumerate(roi_names):
                if i == j:
                    continue
                te_val = transfer_entropy_pair(
                    roi_signals[src][ep], roi_signals[tgt][ep],
                    LOCKED["te_k"], LOCKED["te_bins"]
                )
                te_vals.append(te_val)

                # Track P->C and C->P edges
                if src.startswith("P") and tgt.startswith("C"):
                    te_p2c_vals.append(te_val)
                elif src.startswith("C") and tgt.startswith("P"):
                    te_c2p_vals.append(te_val)

        kappa_epochs[ep] = np.mean(te_vals)
        te_p2c_epochs[ep] = np.mean(te_p2c_vals) if te_p2c_vals else 0
        te_c2p_epochs[ep] = np.mean(te_c2p_vals) if te_c2p_vals else 0

    return {
        "omega_full": omega_full_epochs,
        "omega_df": omega_df_epochs,
        "kappa": kappa_epochs,
        "C_full": omega_full_epochs * kappa_epochs,
        "C_df": omega_df_epochs * kappa_epochs,
        "C_add": omega_full_epochs + kappa_epochs,
        "te_p2c": te_p2c_epochs,
        "te_c2p": te_c2p_epochs,
        "a_pc": te_p2c_epochs - te_c2p_epochs,
    }


# ============================================================
# STATISTICS
# ============================================================

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_sd


def bootstrap_delta_d(a_group1, a_group2, b_group1, b_group2, n_boot, seed):
    """
    Bootstrap delta_d = d(A) - d(B).
    Here groups are NOT paired by subject -- they are epoch collections.
    Resample within each group independently.
    """
    rng = np.random.RandomState(seed)
    n1_a, n2_a = len(a_group1), len(a_group2)
    n1_b, n2_b = len(b_group1), len(b_group2)

    delta_d_boot = []
    for _ in range(n_boot):
        # Resample A
        idx1_a = rng.choice(n1_a, n1_a, replace=True)
        idx2_a = rng.choice(n2_a, n2_a, replace=True)
        d_a = cohens_d(a_group1[idx1_a], a_group2[idx2_a])

        # Resample B
        idx1_b = rng.choice(n1_b, n1_b, replace=True)
        idx2_b = rng.choice(n2_b, n2_b, replace=True)
        d_b = cohens_d(b_group1[idx1_b], b_group2[idx2_b])

        delta_d_boot.append(d_a - d_b)

    delta_d_boot = np.array(delta_d_boot)
    return {
        "delta_d": np.mean(delta_d_boot),
        "ci_low": np.percentile(delta_d_boot, 2.5),
        "ci_high": np.percentile(delta_d_boot, 97.5),
        "p_value": np.mean(delta_d_boot <= 0),
        "passes": np.percentile(delta_d_boot, 2.5) > 0 and np.mean(delta_d_boot <= 0) < LOCKED["alpha_corrected"],
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_subject(edf_path, subject_id):
    """Run TO-CS-02c on one subject."""
    import mne

    # Find scoring file
    subj_dir = os.path.dirname(edf_path)
    txt_files = glob.glob(os.path.join(subj_dir, "*.txt"))
    if not txt_files:
        log(f"  SKIP: No scoring file found")
        return None
    txt_path = txt_files[0]
    log(f"  Scoring file: {os.path.basename(txt_path)}")

    # Load scoring
    scoring = load_scoring(txt_path)
    if not scoring:
        log(f"  SKIP: Empty scoring file")
        return None
    log(f"  Scoring entries: {len(scoring)}")

    # Count N2 epochs in scoring
    n2_count = sum(1 for s in scoring if s == "N2")
    log(f"  N2 scoring epochs: {n2_count}")

    if n2_count < LOCKED["min_epochs_n2_total"] // 3:  # need enough 30s epochs
        log(f"  SKIP: Too few N2 epochs in scoring")
        return None

    # Load EEG
    raw, ch_names = load_and_preprocess(edf_path)
    if raw is None:
        return None

    sfreq = raw.info['sfreq']
    log(f"  Channels: {len(ch_names)}, sfreq: {sfreq} Hz")

    # Map ROIs
    roi_mapping, mapped_channels = map_channels_to_rois(ch_names)
    log(f"  ROIs: {list(roi_mapping.keys())} ({sum(len(v) for v in roi_mapping.values())} channels)")

    if len(roi_mapping) < 4:
        log(f"  SKIP: Too few ROIs")
        return None

    # Extract N2 epochs with sub-epoch tracking
    data = raw.get_data()  # (channels, samples)
    scoring_epoch_samples = int(30 * sfreq)  # 30s per scoring epoch
    sub_epoch_samples = int(LOCKED["epoch_length_sec"] * sfreq)  # 10s
    subs_per_scoring = scoring_epoch_samples // sub_epoch_samples  # = 3

    n2_epochs = []
    n2_epoch_meta = []  # (scoring_index, sub_index)

    for score_idx, stage in enumerate(scoring):
        if stage != "N2":
            continue

        base_sample = score_idx * scoring_epoch_samples

        for sub in range(subs_per_scoring):
            start = base_sample + sub * sub_epoch_samples
            end = start + sub_epoch_samples

            if end > data.shape[1]:
                continue

            epoch = data[:, start:end]

            # Quality check
            ptp = np.ptp(epoch, axis=1) * 1e6
            if np.max(ptp) <= LOCKED["epoch_reject_uv"]:
                n2_epochs.append(epoch)
                n2_epoch_meta.append((score_idx, sub))

    n2_epochs = np.array(n2_epochs) if n2_epochs else np.array([])
    log(f"  Clean N2 10s epochs: {len(n2_epochs)}")

    if len(n2_epochs) < LOCKED["min_epochs_n2_total"]:
        log(f"  SKIP: Too few clean N2 epochs ({len(n2_epochs)} < {LOCKED['min_epochs_n2_total']})")
        return None

    # Classify N2 epochs
    classification = classify_n2_epochs(scoring, n2_epoch_meta)
    n_pre = len(classification["pre_arousal"])
    n_stable = len(classification["stable"])
    log(f"  Pre-arousal: {n_pre}, Stable-N2: {n_stable}, Other-N2: {len(classification['other'])}")

    if n_pre < LOCKED["min_epochs_pre_arousal"]:
        log(f"  SKIP: Too few pre-arousal epochs ({n_pre} < {LOCKED['min_epochs_pre_arousal']})")
        return None
    if n_stable < LOCKED["min_epochs_stable_n2"]:
        log(f"  SKIP: Too few stable-N2 epochs ({n_stable} < {LOCKED['min_epochs_stable_n2']})")
        return None

    # Compute ROI signals for all N2 epochs
    roi_signals = {}
    for roi_name, ch_indices in roi_mapping.items():
        roi_signals[roi_name] = np.mean(n2_epochs[:, ch_indices, :], axis=1)

    # Detect spindle and K-complex epochs
    spindle_epochs = detect_spindle_epochs(n2_epochs, sfreq, roi_signals)
    kc_epochs = detect_kcomplex_epochs(n2_epochs, sfreq, roi_mapping, n2_epochs)
    log(f"  Spindle epochs: {len(spindle_epochs)}/{len(n2_epochs)}")
    log(f"  K-complex epochs: {len(kc_epochs)}/{len(n2_epochs)}")

    # Compute per-epoch measures
    measures = compute_epoch_measures(roi_signals, sfreq)

    # Within-N2 correlation
    corr_within, corr_p = stats.pearsonr(measures["omega_full"], measures["kappa"])
    corr_df_within, _ = stats.pearsonr(measures["omega_df"], measures["kappa"])
    log(f"  Within-N2 corr(Omega_full, kappa) = {corr_within:.3f} (p={corr_p:.4f})")
    log(f"  Within-N2 corr(Omega_4-40, kappa) = {corr_df_within:.3f}")

    # Compile results
    result = {
        "subject_id": subject_id,
        "n_n2_epochs": len(n2_epochs),
        "n_pre_arousal": n_pre,
        "n_stable_n2": n_stable,
        "n_spindle": len(spindle_epochs),
        "n_kcomplex": len(kc_epochs),
        "corr_within_n2_full": corr_within,
        "corr_within_n2_df": corr_df_within,
        "classification": {k: v for k, v in classification.items()},
        "spindle_indices": spindle_epochs,
        "kcomplex_indices": kc_epochs,
        "measures": {k: v.tolist() for k, v in measures.items()},
    }

    # Report group means
    for group_name, indices in [("pre_arousal", classification["pre_arousal"]),
                                  ("stable_n2", classification["stable"]),
                                  ("spindle", spindle_epochs),
                                  ("non_spindle", [i for i in range(len(n2_epochs)) if i not in spindle_epochs])]:
        if len(indices) < 3:
            continue
        idx = np.array(indices)
        log(f"    {group_name} (N={len(idx)}): "
            f"Omega_full={np.mean(measures['omega_full'][idx]):.4f}, "
            f"Omega_4-40={np.mean(measures['omega_df'][idx]):.4f}, "
            f"kappa={np.mean(measures['kappa'][idx]):.4f}, "
            f"C_full={np.mean(measures['C_full'][idx]):.4f}")

    return result


def score_test(all_results):
    """Apply TO-CS-02c decision tree."""
    log(f"\n{'='*60}")
    log("SCORING -- TO-CS-02c DECISION TREE")
    log(f"{'='*60}")

    results = [r for r in all_results if r is not None]
    n = len(results)
    log(f"\nUsable subjects: {n}")

    # Step 0: Minimum N
    if n < LOCKED["min_subjects"]:
        log(f"STOP: N={n} < {LOCKED['min_subjects']} -> INDETERMINATE (I)")
        return "I", {"reason": f"N={n} below minimum"}

    # Step 1: Within-N2 stop rule
    log(f"\n--- Step 1: Within-N2 dissociation check ---")
    within_corrs = [r["corr_within_n2_full"] for r in results]
    median_corr = np.median(within_corrs)
    log(f"  Per-subject within-N2 corr(Omega, kappa):")
    for r in results:
        log(f"    {r['subject_id']}: {r['corr_within_n2_full']:.3f}")
    log(f"  Median: {median_corr:.3f} (threshold: {LOCKED['stop_rule_within_n2_corr']})")

    if median_corr > LOCKED["stop_rule_within_n2_corr"]:
        log(f"  STOP: Median within-N2 corr = {median_corr:.3f} > {LOCKED['stop_rule_within_n2_corr']} -> I")
        return "I", {"reason": f"Median within-N2 corr = {median_corr:.3f}", "per_subject_corrs": within_corrs}

    log(f"  PASS: Dissociation exists within N2")

    # Collect pooled epoch data for pre-arousal vs stable
    log(f"\n--- Step 2: H2 -- Product vs components ---")

    pre_arousal_data = {k: [] for k in ["omega_full", "omega_df", "kappa", "C_full", "C_df", "C_add"]}
    stable_data = {k: [] for k in pre_arousal_data}

    for r in results:
        measures = {k: np.array(v) for k, v in r["measures"].items()}
        pre_idx = np.array(r["classification"]["pre_arousal"])
        stable_idx = np.array(r["classification"]["stable"])

        for key in pre_arousal_data:
            pre_arousal_data[key].extend(measures[key][pre_idx].tolist())
            stable_data[key].extend(measures[key][stable_idx].tolist())

    for key in pre_arousal_data:
        pre_arousal_data[key] = np.array(pre_arousal_data[key])
        stable_data[key] = np.array(stable_data[key])

    n_pre = len(pre_arousal_data["omega_full"])
    n_stable = len(stable_data["omega_full"])
    log(f"  Pooled: {n_pre} pre-arousal, {n_stable} stable-N2 epochs")

    # Effect sizes
    log(f"\n  Effect sizes (pre-arousal vs stable-N2):")
    effect_sizes = {}
    for key in ["omega_full", "omega_df", "kappa", "C_full", "C_df", "C_add"]:
        d = cohens_d(pre_arousal_data[key], stable_data[key])
        effect_sizes[key] = d
        log(f"    d({key}) = {d:.3f} "
            f"(pre={np.mean(pre_arousal_data[key]):.4f}, stable={np.mean(stable_data[key]):.4f})")

    # Comparison (a): C_df vs omega_df
    log(f"\n  Comparison (a): C(4-40) vs Omega(4-40)")
    comp_a = bootstrap_delta_d(
        pre_arousal_data["C_df"], stable_data["C_df"],
        pre_arousal_data["omega_df"], stable_data["omega_df"],
        LOCKED["n_bootstrap"], LOCKED["bootstrap_seed"]
    )
    log(f"    delta_d = {comp_a['delta_d']:.3f}, CI = [{comp_a['ci_low']:.3f}, {comp_a['ci_high']:.3f}], p = {comp_a['p_value']:.4f}")

    # Comparison (b): C_df vs kappa
    log(f"\n  Comparison (b): C(4-40) vs kappa")
    comp_b = bootstrap_delta_d(
        pre_arousal_data["C_df"], stable_data["C_df"],
        pre_arousal_data["kappa"], stable_data["kappa"],
        LOCKED["n_bootstrap"], LOCKED["bootstrap_seed"]
    )
    log(f"    delta_d = {comp_b['delta_d']:.3f}, CI = [{comp_b['ci_low']:.3f}, {comp_b['ci_high']:.3f}], p = {comp_b['p_value']:.4f}")

    # Comparison (c): C_full vs omega_full
    log(f"\n  Comparison (c): C(full) vs Omega(full)")
    comp_c = bootstrap_delta_d(
        pre_arousal_data["C_full"], stable_data["C_full"],
        pre_arousal_data["omega_full"], stable_data["omega_full"],
        LOCKED["n_bootstrap"], LOCKED["bootstrap_seed"]
    )
    log(f"    delta_d = {comp_c['delta_d']:.3f}, CI = [{comp_c['ci_low']:.3f}, {comp_c['ci_high']:.3f}], p = {comp_c['p_value']:.4f}")

    # Comparison (d): C_mult vs C_add
    log(f"\n  Comparison (d): C_mult vs C_add")
    comp_d = bootstrap_delta_d(
        pre_arousal_data["C_full"], stable_data["C_full"],
        pre_arousal_data["C_add"], stable_data["C_add"],
        LOCKED["n_bootstrap"], LOCKED["bootstrap_seed"]
    )
    log(f"    delta_d = {comp_d['delta_d']:.3f}, CI = [{comp_d['ci_low']:.3f}, {comp_d['ci_high']:.3f}], p = {comp_d['p_value']:.4f}")

    # H3: Spindle dissociation
    log(f"\n--- Step 3: H3 -- Spindle dissociation ---")
    spindle_omega = []
    spindle_kappa = []
    non_spindle_omega = []
    non_spindle_kappa = []

    for r in results:
        measures = {k: np.array(v) for k, v in r["measures"].items()}
        sp_idx = r["spindle_indices"]
        non_sp_idx = [i for i in range(r["n_n2_epochs"]) if i not in sp_idx]

        if len(sp_idx) >= 3 and len(non_sp_idx) >= 3:
            spindle_omega.extend(measures["omega_full"][sp_idx].tolist())
            spindle_kappa.extend(measures["kappa"][sp_idx].tolist())
            non_spindle_omega.extend(measures["omega_full"][non_sp_idx].tolist())
            non_spindle_kappa.extend(measures["kappa"][non_sp_idx].tolist())

    h3_confirmed = False
    if spindle_omega:
        d_omega_sp = cohens_d(np.array(spindle_omega), np.array(non_spindle_omega))
        d_kappa_sp = cohens_d(np.array(spindle_kappa), np.array(non_spindle_kappa))

        # Dissociation = different direction or magnitude
        omega_dir = np.sign(d_omega_sp)
        kappa_dir = np.sign(d_kappa_sp)

        log(f"  Spindle vs non-spindle: d(Omega) = {d_omega_sp:.3f}, d(kappa) = {d_kappa_sp:.3f}")
        up_arrow = "up"
        down_arrow = "down"
        eq_arrow = "="
        log(f"  Omega direction: {up_arrow if omega_dir > 0 else down_arrow if omega_dir < 0 else eq_arrow}, "
            f"kappa direction: {up_arrow if kappa_dir > 0 else down_arrow if kappa_dir < 0 else eq_arrow}")

        if omega_dir != kappa_dir and abs(d_omega_sp) > 0.1 and abs(d_kappa_sp) > 0.1:
            log(f"  H3 CONFIRMED: Opposite direction dissociation")
            h3_confirmed = True
        elif abs(d_omega_sp - d_kappa_sp) > 0.3:
            log(f"  H3 CONFIRMED: Different magnitude (delta_d = {abs(d_omega_sp - d_kappa_sp):.3f})")
            h3_confirmed = True
        else:
            log(f"  H3 NOT confirmed: No clear dissociation")

    # H4: K-complex (exploratory)
    log(f"\n--- H4: K-complex analysis (exploratory) ---")
    for r in results:
        measures = {k: np.array(v) for k, v in r["measures"].items()}
        kc_idx = r["kcomplex_indices"]
        non_kc_idx = [i for i in range(r["n_n2_epochs"]) if i not in kc_idx]

        if len(kc_idx) >= 3:
            log(f"  {r['subject_id']}: KC(N={len(kc_idx)}): "
                f"Omega={np.mean(measures['omega_full'][kc_idx]):.4f}, "
                f"kappa={np.mean(measures['kappa'][kc_idx]):.4f} | "
                f"non-KC: Omega={np.mean(measures['omega_full'][non_kc_idx]):.4f}, "
                f"kappa={np.mean(measures['kappa'][non_kc_idx]):.4f}")

    # Apply decision tree
    log(f"\n--- DECISION ---")

    c_beats_omega = comp_a["passes"]  # C_df vs omega_df
    c_beats_kappa = comp_b["passes"]  # C_df vs kappa
    c_beats_full = comp_c["passes"]   # C_full vs omega_full

    if not c_beats_omega and not c_beats_kappa:
        if not h3_confirmed:
            score = "F2"
            log(f"  C doesn't beat components AND no spindle dissociation -> F2")
        else:
            score = "F1"
            log(f"  C doesn't beat components -> F1")
    elif c_beats_omega and c_beats_kappa:
        delta_d_c = comp_c["delta_d"]
        if delta_d_c >= 0.5 and h3_confirmed:
            score = "C3"
            log(f"  C beats both, delta_d>=0.5, H3 confirmed -> C3")
        elif delta_d_c >= 0.5 or h3_confirmed:
            score = "C2"
            log(f"  C beats both -> C2")
        else:
            score = "C2"
            log(f"  C beats both -> C2")
    else:
        score = "C1"
        log(f"  C beats one component -> C1")

    details = {
        "median_within_n2_corr": median_corr,
        "per_subject_corrs": within_corrs,
        "effect_sizes": effect_sizes,
        "comp_a": comp_a,
        "comp_b": comp_b,
        "comp_c": comp_c,
        "comp_d": comp_d,
        "h3_confirmed": h3_confirmed,
        "n_pre_arousal_pooled": n_pre,
        "n_stable_pooled": n_stable,
    }

    return score, details


def main():
    log("=" * 60)
    log("TO-CS-02c: Within-N2 Micro-Arousal Dissociation Test")
    log("Pre-registered: 2026-02-08 | Parameters: LOCKED")
    log("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    edf_files = find_edf_files(DATA_DIR)
    log(f"\nFound {len(edf_files)} EDF files")

    if not edf_files:
        log("ERROR: No EDF files found. Run TO-CS-02b first to extract data.")
        sys.exit(1)

    all_results = []
    for i, edf_path in enumerate(edf_files):
        subject_id = Path(edf_path).stem
        log(f"\n{'='*40}")
        log(f"SUBJECT {i+1}/{len(edf_files)}: {subject_id}")
        log(f"{'='*40}")

        result = run_subject(edf_path, subject_id)
        all_results.append(result)

        if result is not None:
            result_file = os.path.join(RESULTS_DIR, f"{subject_id}_02c.json")
            # Save without the large measure arrays to keep file manageable
            save_result = {k: v for k, v in result.items() if k != "measures"}
            save_result["measures_summary"] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in result["measures"].items()
            }
            with open(result_file, 'w') as f:
                json.dump(save_result, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    usable = [r for r in all_results if r is not None]
    log(f"\n{'='*60}")
    log(f"PIPELINE COMPLETE: {len(usable)}/{len(all_results)} subjects usable")

    score, details = score_test(usable)

    # Save summary
    summary = {
        "test_id": "TO-CS-02c",
        "date_executed": datetime.now().isoformat(),
        "n_subjects": len(usable),
        "score": score,
        "details": {k: v for k, v in details.items()
                    if not isinstance(v, (np.ndarray,))},
    }

    summary_file = os.path.join(RESULTS_DIR, "TO-CS-02c_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    log(f"\n{'='*60}")
    log(f"FINAL SCORE: {score}")
    log(f"Summary saved: {summary_file}")
    log(f"{'='*60}")

    return score


if __name__ == "__main__":
    main()
