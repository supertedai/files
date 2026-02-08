#!/usr/bin/env python3
"""
TO-CS-02b: Multi-Subject Sleep Replication of EFC-C Product Hypothesis
======================================================================
Self-contained pipeline. Run locally on ANPHY-Sleep EDF files.

Pre-registered: 2026-02-08
Parameters: LOCKED (do not modify)

Usage:
    1. Place ANPHY-Sleep ZIP files in a folder
    2. Edit DATA_DIR below to point to that folder
    3. Run: python to_cs_02b_pipeline.py
    4. Results saved to results/ subfolder

Requirements:
    pip install mne numpy scipy pyinform pandas matplotlib seaborn
"""

import os
import sys
import json
import glob
import zipfile
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.signal import welch

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================
# CONFIGURATION — EDIT ONLY THIS SECTION
# ============================================================

# Point this to the folder containing your ANPHY-Sleep ZIP/EDF files
DATA_DIR = "./anphy_sleep_data"

# Output directory
RESULTS_DIR = "./results"

# If ZIPs are not yet extracted, set to True
EXTRACT_ZIPS = True

# ============================================================
# LOCKED PARAMETERS — DO NOT MODIFY (pre-registered 2026-02-08)
# ============================================================

LOCKED = {
    # Epochs
    "epoch_length_sec": 10,
    "epoch_overlap": 0,  # non-overlapping
    "epoch_reject_uv": 100,  # reject if peak-to-peak > 100 uV in any channel
    "min_epochs_per_stage": 10,

    # Spectral entropy
    "se_nperseg": 2048,  # Welch window (if sfreq allows; else sfreq*2)
    "se_noverlap_frac": 0.5,

    # Frequency bands
    "band_full": (0.5, 45.0),
    "band_delta_free": (4.0, 40.0),

    # Transfer entropy (pyinform via wrapper)
    "te_k": 3,  # history length
    "te_bins": 6,  # discretisation bins

    # Bootstrap
    "n_bootstrap": 10000,
    "bootstrap_seed": 42,

    # Multiple comparison
    "alpha": 0.05,
    "n_comparisons": 4,
    "alpha_corrected": 0.05 / 4,  # = 0.0125

    # Minimum subjects
    "min_subjects": 5,
}

# Sleep stage mapping for ANPHY-Sleep .txt scoring files
# Format: STAGE ONSET_SEC DURATION_SEC (tab-separated, 30s epochs)
# "L" = Lights off / setup period (skipped)
# "R" = REM
STAGE_MAP = {
    "W": "Wake", "R": "REM", "N1": "N1", "N2": "N2", "N3": "N3",
    "REM": "REM", "WAKE": "Wake", "wake": "Wake", "rem": "REM",
    "n1": "N1", "n2": "N2", "n3": "N3",
    # "L" (lights/setup) is intentionally omitted -> skipped
}

STAGES_REQUIRED = ["Wake", "N3"]
STAGES_OPTIONAL = ["REM", "N2"]
STAGES_ALL = ["Wake", "REM", "N2", "N3"]


# ============================================================
# ROI DEFINITIONS FOR 10-10 MONTAGE (locked after channel detection)
# ============================================================

# ROI definitions include both 10-10 and old 10-20 names
# (T3=T7, T4=T8, T5=P7, T6=P8 in old nomenclature)
ROI_DEFINITIONS_10_10 = {
    "F_L":  ["Fp1", "AF3", "AF7", "F3", "F5", "F7", "F1"],
    "F_R":  ["Fp2", "AF4", "AF8", "F4", "F6", "F8", "F2", "Fz", "FZ"],
    "C_L":  ["C3", "C5", "C1", "FC3", "FC5", "FC1", "CP3", "CP1", "T7", "T3"],
    "C_R":  ["C4", "C6", "C2", "FC4", "FC6", "FC2", "CP4", "CP2", "Cz", "CZ", "T8", "T4"],
    "P_L":  ["P3", "P5", "P7", "T5", "P1", "CP5", "PO3", "PO7", "TP7"],
    "P_R":  ["P4", "P6", "P8", "T6", "P2", "CP6", "PO4", "PO8", "Pz", "PZ", "TP8"],
    "O_L":  ["O1", "PO7"],
    "O_R":  ["O2", "PO8", "Oz", "OZ"],
}

# Channels to exclude (non-EEG)
EXCLUDE_PATTERNS = [
    "EOG", "EMG", "ECG", "EKG", "eog", "emg", "ecg",
    "Status", "STI", "Trigger", "Event", "DC", "EDF",
    "chin", "Chin", "ChEMG",
    "leg", "Leg", "RLEG", "LLEG",
    "A1", "A2", "M1", "M2",
    "HEOG", "VEOG", "Resp", "SpO2", "Pulse", "Snore",
    "Flow", "Thor", "Abdo", "Position", "Light", "Temp",
    "ZY1", "ZY2", "SO1", "SO2",  # zygomatic / supra-orbital (non-EEG)
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def log(msg):
    """Timestamped logging."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def extract_zips(data_dir):
    """Extract all ZIP files in data_dir."""
    zips = glob.glob(os.path.join(data_dir, "*.zip"))
    for z in zips:
        log(f"Extracting {os.path.basename(z)}...")
        with zipfile.ZipFile(z, 'r') as zf:
            zf.extractall(data_dir)
    return len(zips)


def find_edf_files(data_dir):
    """Find all EDF/EDF+ files recursively."""
    patterns = ["**/*.edf", "**/*.EDF", "**/*.edf+", "**/*.rec", "**/*.REC"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(data_dir, p), recursive=True))
    return sorted(set(files))


def identify_subjects(edf_files):
    """
    Group EDF files by subject.
    ANPHY-Sleep typically has one EDF per subject (whole-night recording).
    Returns dict: subject_id -> filepath
    """
    subjects = {}
    for f in edf_files:
        basename = os.path.basename(f)
        # Try to extract subject ID from filename
        name = Path(f).stem
        # Remove common suffixes
        for suffix in ["-PSG", "_PSG", "_psg", "-psg", "_eeg", "-eeg", "_EEG"]:
            name = name.replace(suffix, "")
        subjects[name] = f
    return subjects


def is_eeg_channel(ch_name):
    """Check if channel name looks like EEG (not EOG/EMG/etc)."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern.lower() in ch_name.lower():
            return False
    return True


def normalise_channel_name(name):
    """
    Normalise channel name for ROI matching.
    Strips '-Ref' suffix (EPCTL21 format), removes dots/spaces, lowercases.
    'Fp1-Ref' -> 'fp1', 'FZ-Ref' -> 'fz', 'CP5' -> 'cp5'
    """
    n = name.strip()
    # Strip common reference suffixes
    for suffix in ["-Ref", "-ref", "-REF", "-Avg", "-avg", "-AVG"]:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
    return n


def map_channels_to_rois(ch_names):
    """
    Map available EEG channels to ROIs.
    Handles both 'Fp1' and 'Fp1-Ref' naming conventions.
    Returns: dict roi_name -> list of channel indices
    """
    # Build lookup: normalised name -> channel index
    ch_norm = {normalise_channel_name(ch).lower(): i for i, ch in enumerate(ch_names)}

    roi_mapping = {}
    mapped_channels = {}

    for roi_name, roi_channels in ROI_DEFINITIONS_10_10.items():
        indices = []
        matched = []
        for rc in roi_channels:
            rc_lower = rc.lower()
            if rc_lower in ch_norm:
                indices.append(ch_norm[rc_lower])
                matched.append(ch_names[ch_norm[rc_lower]])

        if len(indices) >= 2:
            roi_mapping[roi_name] = indices
            mapped_channels[roi_name] = matched
        else:
            log(f"  WARNING: ROI {roi_name} has only {len(indices)} channels -- excluded")

    return roi_mapping, mapped_channels


def load_and_preprocess(filepath, target_sfreq=250):
    """
    Load EDF, extract EEG channels, bandpass filter, resample.
    Returns: raw MNE object, channel names
    """
    import mne
    mne.set_log_level('WARNING')

    log(f"  Loading {os.path.basename(filepath)}...")

    try:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    except Exception as e:
        log(f"  ERROR loading {filepath}: {e}")
        return None, None

    all_ch = raw.ch_names
    log(f"  Channels found: {len(all_ch)}, sfreq: {raw.info['sfreq']} Hz")

    eeg_picks = [ch for ch in all_ch if is_eeg_channel(ch)]
    if len(eeg_picks) < 10:
        log(f"  WARNING: Only {len(eeg_picks)} EEG channels found. Trying all non-status channels.")
        eeg_picks = [ch for ch in all_ch if ch.lower() not in ['status', 'sti 014', 'event']]

    raw.pick_channels(eeg_picks)
    log(f"  EEG channels selected: {len(raw.ch_names)}")

    if raw.info['sfreq'] != target_sfreq:
        log(f"  Resampling {raw.info['sfreq']} -> {target_sfreq} Hz")
        raw.resample(target_sfreq)

    raw.filter(0.5, 45.0, verbose=False)

    return raw, raw.ch_names


def load_scoring_txt(txt_path):
    """
    Load sleep stage scoring from ANPHY-Sleep .txt file.
    Format: STAGE<tab>ONSET_SEC<tab>DURATION_SEC  (one line per 30s epoch)
    Example: "W  450  30", "N2  12870  30", "L  0  30"
    Returns: list of (onset_sec, duration_sec, stage_label)
    """
    stages = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            label = parts[0]
            try:
                onset = float(parts[1])
                duration = float(parts[2])
            except ValueError:
                continue

            stage = STAGE_MAP.get(label, None)
            if stage is not None:
                stages.append((onset, duration, stage))

    return stages


def find_scoring_file(edf_path):
    """
    Find the .txt scoring file matching an EDF file.
    Looks for SUBJECTID.txt in the same directory as the EDF.
    """
    edf_dir = os.path.dirname(edf_path)
    stem = Path(edf_path).stem  # e.g. "EPCTL21"

    # Try same directory
    txt_path = os.path.join(edf_dir, stem + ".txt")
    if os.path.exists(txt_path):
        return txt_path

    # Try parent directory
    txt_path = os.path.join(os.path.dirname(edf_dir), stem + ".txt")
    if os.path.exists(txt_path):
        return txt_path

    # Search recursively from DATA_DIR
    matches = glob.glob(os.path.join(DATA_DIR, "**", stem + ".txt"), recursive=True)
    if matches:
        return matches[0]

    return None


def epoch_by_stage(raw, annotations, epoch_len, reject_uv):
    """
    Cut continuous data into fixed-length epochs, labelled by sleep stage.
    Returns: dict stage -> np.array (n_epochs, n_channels, n_samples)
    """
    sfreq = raw.info['sfreq']
    n_samples_epoch = int(epoch_len * sfreq)
    data = raw.get_data()
    n_channels = data.shape[0]

    stage_epochs = {s: [] for s in STAGES_ALL}

    for onset, duration, stage in annotations:
        if stage not in STAGES_ALL:
            continue

        start_sample = int(onset * sfreq)
        end_sample = int((onset + duration) * sfreq)

        pos = start_sample
        while pos + n_samples_epoch <= end_sample and pos + n_samples_epoch <= data.shape[1]:
            epoch = data[:, pos:pos + n_samples_epoch]

            ptp = np.ptp(epoch, axis=1) * 1e6
            if np.max(ptp) <= reject_uv:
                stage_epochs[stage].append(epoch)

            pos += n_samples_epoch

    result = {}
    for stage in STAGES_ALL:
        if len(stage_epochs[stage]) > 0:
            result[stage] = np.array(stage_epochs[stage])
            log(f"    {stage}: {len(stage_epochs[stage])} clean epochs")
        else:
            log(f"    {stage}: 0 epochs")

    return result


def balance_epochs(stage_epochs, stages):
    """Balance epoch counts across stages (use minimum)."""
    rng = np.random.RandomState(LOCKED["bootstrap_seed"])
    counts = {s: stage_epochs[s].shape[0] for s in stages if s in stage_epochs}
    if not counts:
        return stage_epochs

    min_count = min(counts.values())

    balanced = {}
    for s in stages:
        if s in stage_epochs:
            n = stage_epochs[s].shape[0]
            if n > min_count:
                idx = rng.choice(n, min_count, replace=False)
                balanced[s] = stage_epochs[s][idx]
            else:
                balanced[s] = stage_epochs[s]

    return balanced


# ============================================================
# MEASURE COMPUTATIONS (LOCKED)
# ============================================================

def compute_roi_signals(epochs, roi_mapping):
    """
    Average channels within each ROI.
    Input: (n_epochs, n_channels, n_samples)
    Output: dict roi_name -> (n_epochs, n_samples)
    """
    roi_signals = {}
    for roi_name, ch_indices in roi_mapping.items():
        roi_signals[roi_name] = np.mean(epochs[:, ch_indices, :], axis=1)
    return roi_signals


def spectral_entropy(signal, sfreq, band):
    """
    Compute normalised spectral entropy (Welch PSD) for a single signal.
    """
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
    if H_max == 0:
        return 0.0

    return H / H_max


def compute_omega(roi_signals, sfreq, band):
    """Compute Omega averaged over all ROIs."""
    n_epochs = list(roi_signals.values())[0].shape[0]
    omega_per_epoch = []

    for ep in range(n_epochs):
        se_values = []
        for roi_name, signals in roi_signals.items():
            se = spectral_entropy(signals[ep], sfreq, band)
            se_values.append(se)
        omega_per_epoch.append(np.mean(se_values))

    return np.mean(omega_per_epoch), np.array(omega_per_epoch)


def compute_omega_per_roi(roi_signals, sfreq, band):
    """Compute Omega per ROI (not averaged)."""
    result = {}
    n_epochs = list(roi_signals.values())[0].shape[0]

    for roi_name, signals in roi_signals.items():
        se_epochs = []
        for ep in range(n_epochs):
            se = spectral_entropy(signals[ep], sfreq, band)
            se_epochs.append(se)
        result[roi_name] = np.mean(se_epochs)

    return result


def discretise(signal, n_bins):
    """Discretise continuous signal into n_bins equal-width bins."""
    sig_min, sig_max = signal.min(), signal.max()
    if sig_max == sig_min:
        return np.zeros(len(signal), dtype=int)
    bins = np.linspace(sig_min, sig_max, n_bins + 1)
    digitised = np.digitize(signal, bins[1:-1])
    return digitised.astype(int)


def transfer_entropy_pair(source, target, k, n_bins):
    """Compute transfer entropy: TE(source -> target)."""
    try:
        from pyinform import transfer_entropy as te_func
        src_d = discretise(source, n_bins)
        tgt_d = discretise(target, n_bins)
        te_val = te_func(src_d, tgt_d, k=k)
        return te_val
    except ImportError:
        return _manual_te(source, target, k, n_bins)
    except Exception:
        return 0.0


def _manual_te(source, target, k, n_bins):
    """Manual transfer entropy computation (fallback if pyinform unavailable)."""
    src_d = discretise(source, n_bins)
    tgt_d = discretise(target, n_bins)
    n = len(src_d)

    if n <= k + 1:
        return 0.0

    from collections import Counter

    joint_xyz = Counter()
    joint_yz = Counter()
    joint_yz_only = Counter()
    joint_y = Counter()

    for t in range(k, n - 1):
        y_future = tgt_d[t + 1]
        y_past = tuple(tgt_d[t - k + 1:t + 1])
        x_past = tuple(src_d[t - k + 1:t + 1])

        joint_xyz[(y_future, y_past, x_past)] += 1
        joint_yz[(y_future, y_past)] += 1
        joint_yz_only[y_past] += 1
        joint_y[(y_past, x_past)] += 1

    total = sum(joint_xyz.values())
    if total == 0:
        return 0.0

    te = 0.0
    for (yf, yp, xp), count in joint_xyz.items():
        p_xyz = count / total
        p_yf_given_ypxp = count / joint_y[(yp, xp)] if joint_y[(yp, xp)] > 0 else 0
        p_yf_given_yp = joint_yz[(yf, yp)] / joint_yz_only[yp] if joint_yz_only[yp] > 0 else 0

        if p_yf_given_ypxp > 0 and p_yf_given_yp > 0:
            te += p_xyz * np.log2(p_yf_given_ypxp / p_yf_given_yp)

    return max(te, 0.0)


def compute_kappa(roi_signals, k, n_bins):
    """Compute kappa = mean TE over all directed ROI pairs."""
    roi_names = list(roi_signals.keys())
    n_rois = len(roi_names)
    n_epochs = roi_signals[roi_names[0]].shape[0]

    te_matrix = np.zeros((n_rois, n_rois))

    for i, src_name in enumerate(roi_names):
        for j, tgt_name in enumerate(roi_names):
            if i == j:
                continue
            te_epochs = []
            for ep in range(n_epochs):
                te_val = transfer_entropy_pair(
                    roi_signals[src_name][ep],
                    roi_signals[tgt_name][ep],
                    k, n_bins
                )
                te_epochs.append(te_val)
            te_matrix[i, j] = np.mean(te_epochs)

    mask = ~np.eye(n_rois, dtype=bool)
    kappa_global = te_matrix[mask].mean()

    def get_te(src, tgt):
        if src in roi_names and tgt in roi_names:
            i = roi_names.index(src)
            j = roi_names.index(tgt)
            return te_matrix[i, j]
        return np.nan

    edges = {
        "TE_P2C": np.mean([get_te("P_L", "C_L"), get_te("P_L", "C_R"),
                           get_te("P_R", "C_L"), get_te("P_R", "C_R")]),
        "TE_C2P": np.mean([get_te("C_L", "P_L"), get_te("C_L", "P_R"),
                           get_te("C_R", "P_L"), get_te("C_R", "P_R")]),
        "TE_F2P": np.mean([get_te("F_L", "P_L"), get_te("F_L", "P_R"),
                           get_te("F_R", "P_L"), get_te("F_R", "P_R")]),
        "TE_P2F": np.mean([get_te("P_L", "F_L"), get_te("P_L", "F_R"),
                           get_te("P_R", "F_L"), get_te("P_R", "F_R")]),
    }

    edges["A_PC"] = edges["TE_P2C"] - edges["TE_C2P"]
    edges["A_FP"] = edges["TE_F2P"] - edges["TE_P2F"]

    if "P_L" in roi_names:
        i_pl = roi_names.index("P_L")
        kappa_in_pl = np.mean([te_matrix[j, i_pl] for j in range(n_rois) if j != i_pl])
        kappa_out_pl = np.mean([te_matrix[i_pl, j] for j in range(n_rois) if j != i_pl])
        edges["kappa_hub_PL"] = np.mean([kappa_in_pl, kappa_out_pl])
    else:
        edges["kappa_hub_PL"] = np.nan

    return kappa_global, te_matrix, roi_names, edges


def compute_kappa_per_epoch(roi_signals, k, n_bins):
    """Compute kappa per epoch (for correlation with Omega)."""
    roi_names = list(roi_signals.keys())
    n_rois = len(roi_names)
    n_epochs = roi_signals[roi_names[0]].shape[0]

    kappa_epochs = []
    for ep in range(n_epochs):
        te_vals = []
        for i, src in enumerate(roi_names):
            for j, tgt in enumerate(roi_names):
                if i == j:
                    continue
                te_val = transfer_entropy_pair(
                    roi_signals[src][ep],
                    roi_signals[tgt][ep],
                    k, n_bins
                )
                te_vals.append(te_val)
        kappa_epochs.append(np.mean(te_vals))

    return np.array(kappa_epochs)


# ============================================================
# STATISTICAL TESTS (LOCKED)
# ============================================================

def cohens_d(group1, group2):
    """Cohen's d (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_sd


def bootstrap_delta_d(measure_a_w, measure_a_n3, measure_b_w, measure_b_n3, n_boot, seed):
    """Bootstrap delta_d = d(A) - d(B) with CI. Paired resampling."""
    rng = np.random.RandomState(seed)
    n = len(measure_a_w)
    assert n == len(measure_a_n3) == len(measure_b_w) == len(measure_b_n3)

    delta_d_boot = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        d_a = cohens_d(measure_a_w[idx], measure_a_n3[idx])
        d_b = cohens_d(measure_b_w[idx], measure_b_n3[idx])
        delta_d_boot.append(d_a - d_b)

    delta_d_boot = np.array(delta_d_boot)
    ci_low = np.percentile(delta_d_boot, 2.5)
    ci_high = np.percentile(delta_d_boot, 97.5)
    p_value = np.mean(delta_d_boot <= 0)

    return {
        "delta_d": np.mean(delta_d_boot),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "passes": ci_low > 0 and p_value < LOCKED["alpha_corrected"],
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_subject(filepath, subject_id):
    """Run full pipeline on one subject."""
    import mne

    # Find scoring .txt file FIRST (cheap check before loading big EDF)
    scoring_path = find_scoring_file(filepath)
    if scoring_path is None:
        log(f"  SKIP: No scoring .txt file found for {subject_id}")
        return None
    log(f"  Scoring file: {os.path.basename(scoring_path)}")

    raw, ch_names = load_and_preprocess(filepath)
    if raw is None:
        return None

    sfreq = raw.info['sfreq']

    roi_mapping, mapped_channels = map_channels_to_rois(ch_names)
    log(f"  ROIs mapped: {list(roi_mapping.keys())} ({sum(len(v) for v in roi_mapping.values())} channels)")

    if len(roi_mapping) < 4:
        log(f"  SKIP: Too few ROIs ({len(roi_mapping)})")
        return None

    annotations = load_scoring_txt(scoring_path)
    if not annotations:
        log(f"  SKIP: No valid sleep stages in scoring file")
        return None

    log(f"  Scoring entries loaded: {len(annotations)}")
    stage_counts = {}
    for _, _, s in annotations:
        stage_counts[s] = stage_counts.get(s, 0) + 1
    log(f"  Stage distribution: {stage_counts}")

    stage_epochs = epoch_by_stage(
        raw, annotations,
        LOCKED["epoch_length_sec"],
        LOCKED["epoch_reject_uv"]
    )

    for req_stage in STAGES_REQUIRED:
        if req_stage not in stage_epochs or stage_epochs[req_stage].shape[0] < LOCKED["min_epochs_per_stage"]:
            log(f"  SKIP: Insufficient {req_stage} epochs "
                f"({stage_epochs.get(req_stage, np.array([])).shape[0] if req_stage in stage_epochs else 0})")
            return None

    available_stages = [s for s in STAGES_ALL if s in stage_epochs and
                        stage_epochs[s].shape[0] >= LOCKED["min_epochs_per_stage"]]
    stage_epochs = balance_epochs(stage_epochs, available_stages)
    log(f"  Balanced stages: {[(s, stage_epochs[s].shape[0]) for s in available_stages if s in stage_epochs]}")

    results = {
        "subject_id": subject_id,
        "sfreq": sfreq,
        "n_rois": len(roi_mapping),
        "roi_mapping": mapped_channels,
        "stages": {},
    }

    for stage in available_stages:
        if stage not in stage_epochs:
            continue

        epochs = stage_epochs[stage]
        n_ep = epochs.shape[0]
        log(f"  Computing {stage} ({n_ep} epochs)...")

        roi_signals = compute_roi_signals(epochs, roi_mapping)

        omega_full, omega_full_epochs = compute_omega(roi_signals, sfreq, LOCKED["band_full"])
        omega_df, omega_df_epochs = compute_omega(roi_signals, sfreq, LOCKED["band_delta_free"])
        omega_per_roi = compute_omega_per_roi(roi_signals, sfreq, LOCKED["band_full"])

        log(f"    Computing TE ({len(roi_mapping)} ROIs, {n_ep} epochs)...")
        kappa, te_matrix, roi_names, edges = compute_kappa(
            roi_signals, LOCKED["te_k"], LOCKED["te_bins"]
        )

        kappa_epochs = compute_kappa_per_epoch(
            roi_signals, LOCKED["te_k"], LOCKED["te_bins"]
        )

        C_full = omega_full * kappa
        C_df = omega_df * kappa
        C_add = omega_full + kappa

        C_PL = omega_per_roi.get("P_L", np.nan) * edges.get("kappa_hub_PL", np.nan)

        if "O_L" in roi_mapping and "O_L" in roi_names:
            i_ol = roi_names.index("O_L")
            kappa_hub_ol = np.mean([te_matrix[j, i_ol] for j in range(len(roi_names)) if j != i_ol])
            C_OL = omega_per_roi.get("O_L", np.nan) * kappa_hub_ol
        else:
            C_OL = np.nan

        results["stages"][stage] = {
            "n_epochs": n_ep,
            "omega_full": omega_full,
            "omega_delta_free": omega_df,
            "omega_per_roi": omega_per_roi,
            "kappa": kappa,
            "te_matrix": te_matrix.tolist(),
            "te_roi_names": roi_names,
            "edges": edges,
            "C_full": C_full,
            "C_delta_free": C_df,
            "C_additive": C_add,
            "C_PL": C_PL,
            "C_OL": C_OL,
            "omega_full_epochs": omega_full_epochs.tolist(),
            "kappa_epochs": kappa_epochs.tolist(),
        }

        log(f"    Omega_full={omega_full:.4f}, Omega_4-40={omega_df:.4f}, "
            f"kappa={kappa:.4f}, C_full={C_full:.4f}, C_4-40={C_df:.4f}")
        log(f"    TE(P->C)={edges['TE_P2C']:.4f}, TE(C->P)={edges['TE_C2P']:.4f}, "
            f"A_PC={edges['A_PC']:.4f}")

    return results


def score_test(all_results):
    """Apply pre-registered decision tree and scoring criteria."""
    log("\n" + "=" * 60)
    log("SCORING -- TO-CS-02b PRE-REGISTERED DECISION TREE")
    log("=" * 60)

    subjects = [r for r in all_results if r is not None]
    n = len(subjects)
    log(f"\nUsable subjects: {n}")

    if n < LOCKED["min_subjects"]:
        log(f"STOP: N={n} < {LOCKED['min_subjects']} -> INDETERMINATE (I)")
        return "I", {"reason": f"N={n} below minimum {LOCKED['min_subjects']}"}

    measures = {
        "omega_full": {"Wake": [], "N3": []},
        "omega_df": {"Wake": [], "N3": []},
        "kappa": {"Wake": [], "N3": []},
        "C_full": {"Wake": [], "N3": []},
        "C_df": {"Wake": [], "N3": []},
        "C_add": {"Wake": [], "N3": []},
        "C_PL": {"Wake": [], "N3": []},
        "C_OL": {"Wake": [], "N3": []},
        "TE_P2C": {"Wake": [], "N3": []},
        "A_PC": {"Wake": [], "N3": []},
    }

    rem_kappa = {"Wake": [], "REM": []}
    omega_kappa_corr_data = []

    for subj in subjects:
        stages = subj["stages"]
        if "Wake" not in stages or "N3" not in stages:
            continue

        w = stages["Wake"]
        n3 = stages["N3"]

        measures["omega_full"]["Wake"].append(w["omega_full"])
        measures["omega_full"]["N3"].append(n3["omega_full"])
        measures["omega_df"]["Wake"].append(w["omega_delta_free"])
        measures["omega_df"]["N3"].append(n3["omega_delta_free"])
        measures["kappa"]["Wake"].append(w["kappa"])
        measures["kappa"]["N3"].append(n3["kappa"])
        measures["C_full"]["Wake"].append(w["C_full"])
        measures["C_full"]["N3"].append(n3["C_full"])
        measures["C_df"]["Wake"].append(w["C_delta_free"])
        measures["C_df"]["N3"].append(n3["C_delta_free"])
        measures["C_add"]["Wake"].append(w["C_additive"])
        measures["C_add"]["N3"].append(n3["C_additive"])
        measures["C_PL"]["Wake"].append(w["C_PL"])
        measures["C_PL"]["N3"].append(n3["C_PL"])
        measures["C_OL"]["Wake"].append(w["C_OL"])
        measures["C_OL"]["N3"].append(n3["C_OL"])
        measures["TE_P2C"]["Wake"].append(w["edges"]["TE_P2C"])
        measures["TE_P2C"]["N3"].append(n3["edges"]["TE_P2C"])
        measures["A_PC"]["Wake"].append(w["edges"]["A_PC"])
        measures["A_PC"]["N3"].append(n3["edges"]["A_PC"])

        for stage_name in ["Wake", "N3"]:
            if stage_name in stages:
                s = stages[stage_name]
                for o, k in zip(s["omega_full_epochs"], s["kappa_epochs"]):
                    omega_kappa_corr_data.append((o, k))

        if "REM" in stages:
            rem_kappa["Wake"].append(w["kappa"])
            rem_kappa["REM"].append(stages["REM"]["kappa"])

    for m in measures:
        for s in measures[m]:
            measures[m][s] = np.array(measures[m][s])

    n_paired = len(measures["omega_full"]["Wake"])
    log(f"Subjects with both Wake and N3: {n_paired}")

    if n_paired < LOCKED["min_subjects"]:
        log(f"STOP: Only {n_paired} paired subjects -> INDETERMINATE (I)")
        return "I", {"reason": f"Only {n_paired} paired subjects"}

    # Step 1: Dissociation check
    log("\n--- Step 1: Dissociation check ---")
    omega_arr = np.array([x[0] for x in omega_kappa_corr_data])
    kappa_arr = np.array([x[1] for x in omega_kappa_corr_data])
    corr_ok, corr_p = stats.pearsonr(omega_arr, kappa_arr)
    log(f"corr(Omega, kappa) = {corr_ok:.3f} (p={corr_p:.4f})")

    if abs(corr_ok) > 0.8:
        log("STOP: corr > 0.8 -> INDETERMINATE (I)")
        return "I", {"reason": f"corr(Omega,kappa) = {corr_ok:.3f} > 0.8"}

    # Report effect sizes
    log("\n--- Effect sizes (Cohen's d, Wake vs N3) ---")
    effect_sizes = {}
    for m_name in measures:
        w = measures[m_name]["Wake"]
        n3 = measures[m_name]["N3"]
        if len(w) > 0 and len(n3) > 0:
            d = cohens_d(w, n3)
            effect_sizes[m_name] = d
            log(f"  d({m_name}) = {d:.3f}")

    # Step 2: Comparison (a) -- C_df vs omega_df
    log("\n--- Step 2: Comparison (a): C(4-40) x kappa vs Omega(4-40) ---")
    comp_a = bootstrap_delta_d(
        measures["C_df"]["Wake"], measures["C_df"]["N3"],
        measures["omega_df"]["Wake"], measures["omega_df"]["N3"],
        LOCKED["n_bootstrap"], LOCKED["bootstrap_seed"]
    )
    log(f"  delta_d = {comp_a['delta_d']:.3f}, CI = [{comp_a['ci_low']:.3f}, {comp_a['ci_high']:.3f}], "
        f"p = {comp_a['p_value']:.4f}")
    log(f"  Passes: {comp_a['passes']}")

    if not comp_a["passes"]:
        h2_fail = True
        h4_fail = True

        a_pc_wake = measures["A_PC"]["Wake"]
        a_pc_n3 = measures["A_PC"]["N3"]
        if len(a_pc_wake) >= LOCKED["min_subjects"]:
            _, h2_p = stats.wilcoxon(np.abs(a_pc_wake), np.abs(a_pc_n3), alternative='greater')
            h2_fail = h2_p >= 0.05
            log(f"\n  H2 (directional flattening): |A_PC(Wake)| > |A_PC(N3)|, p = {h2_p:.4f}")

        te_p2c_delta = np.mean(measures["TE_P2C"]["Wake"]) - np.mean(measures["TE_P2C"]["N3"])
        log(f"  H4: TE(P->C) Wake-N3 delta = {te_p2c_delta:.4f}")

        if h2_fail and h4_fail:
            score = "F2"
            log(f"\n  Comparison (a) FAILS + H2 absent + H4 uncertain -> SCORE: F2")
        else:
            score = "F1"
            log(f"\n  Comparison (a) FAILS -> SCORE: F1")

        return score, {
            "comp_a": comp_a,
            "effect_sizes": effect_sizes,
            "corr_omega_kappa": corr_ok,
        }

    # Step 3: Comparison (c) -- C_full vs omega_full
    log("\n--- Step 3: Comparison (c): C(full) x kappa vs Omega(full) ---")
    comp_c = bootstrap_delta_d(
        measures["C_full"]["Wake"], measures["C_full"]["N3"],
        measures["omega_full"]["Wake"], measures["omega_full"]["N3"],
        LOCKED["n_bootstrap"], LOCKED["bootstrap_seed"]
    )
    log(f"  delta_d = {comp_c['delta_d']:.3f}, CI = [{comp_c['ci_low']:.3f}, {comp_c['ci_high']:.3f}], "
        f"p = {comp_c['p_value']:.4f}")
    log(f"  Passes: {comp_c['passes']}")

    if not comp_c["passes"]:
        log(f"\n  (a) passes, (c) fails -> SCORE: C1 (replicates propofol finding)")
        return "C1", {
            "comp_a": comp_a,
            "comp_c": comp_c,
            "effect_sizes": effect_sizes,
            "corr_omega_kappa": corr_ok,
        }

    # Step 4: delta_d threshold and H4
    log("\n--- Step 4: delta_d >= 0.5 and cross-paradigm check ---")
    large_delta = comp_c["delta_d"] >= 0.5
    log(f"  delta_d(c) = {comp_c['delta_d']:.3f} {'>=' if large_delta else '<'} 0.5")

    te_p2c_delta = np.mean(measures["TE_P2C"]["Wake"]) - np.mean(measures["TE_P2C"]["N3"])
    log(f"  TE(P->C) delta = {te_p2c_delta:.4f}")

    if large_delta:
        score = "C3"
        log(f"\n  SCORE: C3 (strong support)")
    else:
        score = "C2"
        log(f"\n  SCORE: C2 (moderate support)")

    # Secondary: REM analysis (exploratory)
    log("\n--- Secondary: REM kappa analysis (H3, exploratory) ---")
    if len(rem_kappa["Wake"]) >= 3:
        rem_higher = sum(1 for r, w in zip(rem_kappa["REM"], rem_kappa["Wake"]) if r >= w)
        pct = rem_higher / len(rem_kappa["Wake"]) * 100
        log(f"  REM kappa >= Wake kappa in {rem_higher}/{len(rem_kappa['Wake'])} subjects ({pct:.0f}%)")
        log(f"  NOTE: Artifact controls (S7) not applied in this automated run -- interpret with caution")
    else:
        log(f"  Insufficient REM data ({len(rem_kappa['Wake'])} subjects)")

    # Secondary: Topographic specificity
    log("\n--- Secondary: Topographic specificity ---")
    if "C_PL" in effect_sizes and "C_OL" in effect_sizes:
        log(f"  d(C_PL) = {effect_sizes['C_PL']:.3f}")
        log(f"  d(C_OL) = {effect_sizes['C_OL']:.3f}")
        if effect_sizes["C_PL"] > effect_sizes["C_OL"]:
            log(f"  Parietal > Occipital (confirmed)")
        else:
            log(f"  WARNING: Occipital >= Parietal -- topographic specificity fails")

    return score, {
        "comp_a": comp_a,
        "comp_c": comp_c,
        "effect_sizes": effect_sizes,
        "corr_omega_kappa": corr_ok,
    }


def main():
    """Main execution."""
    log("=" * 60)
    log("TO-CS-02b: Sleep Replication Pipeline")
    log("Pre-registered: 2026-02-08 | Parameters: LOCKED")
    log("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if EXTRACT_ZIPS:
        n_zips = extract_zips(DATA_DIR)
        log(f"Extracted {n_zips} ZIP files")

    edf_files = find_edf_files(DATA_DIR)
    log(f"\nFound {len(edf_files)} EDF files:")
    for f in edf_files:
        log(f"  {f}")

    if not edf_files:
        log("ERROR: No EDF files found. Check DATA_DIR path.")
        sys.exit(1)

    subject_map = identify_subjects(edf_files)
    log(f"\nIdentified {len(subject_map)} subjects:")
    for sid, fpath in subject_map.items():
        log(f"  {sid}: {os.path.basename(fpath)}")

    all_results = []
    for i, (subject_id, filepath) in enumerate(subject_map.items()):
        log(f"\n{'=' * 40}")
        log(f"SUBJECT {i+1}/{len(subject_map)}: {subject_id}")
        log(f"{'=' * 40}")

        result = run_subject(filepath, subject_id)
        all_results.append(result)

        if result is not None:
            result_file = os.path.join(RESULTS_DIR, f"{subject_id}_results.json")
            result_json = json.loads(json.dumps(result, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x))
            with open(result_file, 'w') as f:
                json.dump(result_json, f, indent=2)
            log(f"  Results saved: {result_file}")

    usable = [r for r in all_results if r is not None]
    log(f"\n{'=' * 60}")
    log(f"PIPELINE COMPLETE: {len(usable)}/{len(all_results)} subjects usable")
    log(f"{'=' * 60}")

    score, details = score_test(usable)

    summary = {
        "test_id": "TO-CS-02b",
        "date_executed": datetime.now().isoformat(),
        "pre_registered": "2026-02-08",
        "n_subjects_total": len(subject_map),
        "n_subjects_usable": len(usable),
        "subjects_usable": [r["subject_id"] for r in usable],
        "locked_parameters": LOCKED,
        "score": score,
        "details": {k: v for k, v in details.items() if not isinstance(v, np.ndarray)},
        "per_subject_stages": {
            r["subject_id"]: {
                stage: {
                    "omega_full": s["omega_full"],
                    "omega_delta_free": s["omega_delta_free"],
                    "kappa": s["kappa"],
                    "C_full": s["C_full"],
                    "C_delta_free": s["C_delta_free"],
                    "TE_P2C": s["edges"]["TE_P2C"],
                    "TE_C2P": s["edges"]["TE_C2P"],
                    "A_PC": s["edges"]["A_PC"],
                    "n_epochs": s["n_epochs"],
                }
                for stage, s in r["stages"].items()
            }
            for r in usable
        },
    }

    summary_file = os.path.join(RESULTS_DIR, "TO-CS-02b_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    log(f"\n{'=' * 60}")
    log(f"FINAL SCORE: {score}")
    log(f"Summary saved: {summary_file}")
    log(f"{'=' * 60}")

    log(f"\n{'=' * 60}")
    log("RESULTS TABLE")
    log(f"{'=' * 60}")
    log(f"{'Subject':<12} {'Stage':<6} {'Omega_full':>10} {'Omega_4-40':>10} {'kappa':>8} {'C_full':>8} {'C_4-40':>8} {'TE(P->C)':>9} {'A_PC':>8}")
    log("-" * 85)
    for r in usable:
        for stage in STAGES_ALL:
            if stage in r["stages"]:
                s = r["stages"][stage]
                log(f"{r['subject_id']:<12} {stage:<6} {s['omega_full']:>10.4f} {s['omega_delta_free']:>10.4f} "
                    f"{s['kappa']:>8.4f} {s['C_full']:>8.4f} {s['C_delta_free']:>8.4f} "
                    f"{s['edges']['TE_P2C']:>9.4f} {s['edges']['A_PC']:>8.4f}")

    return score


if __name__ == "__main__":
    score = main()
