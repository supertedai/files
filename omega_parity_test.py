#!/usr/bin/env python3
"""
OMEGA PARITY TEST — Estimator Order Sensitivity
================================================
Purpose: Determine whether the d=1.54 -> 0.23 collapse is explained by
         SE(mean(PSD)) vs mean(SE(PSD)) — i.e., Jensen's inequality.

This script computes THREE Omega variants on the SAME epochs:

  Omega_A = SE(mean(PSD_epoch))     "preview-like" — aggregate PSD first, then entropy
  Omega_B = mean(SE(PSD_epoch))     "pipeline-like" — entropy first, then aggregate
  Omega_C = SE(median(PSD_epoch))   "robust control" — median PSD, then entropy

For each variant:
  - Cohen's d (responsive vs drowsy at moderate sedation)
  - Per-subject values
  - Cross-variant correlations

If Omega_A reproduces d~1.5 and Omega_B reproduces d~0.2, the estimator order
is the explanation.

Usage:
  python omega_parity_test.py --data-dir "sedation-restingstate/Sedation-RestingState"

Requires: numpy, scipy, h5py, mne
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
from collections import defaultdict
from scipy.signal import welch
from scipy import stats
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    import mne
    mne.set_log_level('ERROR')
except ImportError:
    print("ERROR: mne not installed. Run: pip install mne")
    sys.exit(1)

try:
    import h5py
except ImportError:
    print("ERROR: h5py not installed. Run: pip install h5py")
    sys.exit(1)

# ============================================================
# LOCKED PARAMETERS (inherited from run_pipeline.py)
# ============================================================
SE_NPERSEG = 512
BAND_FULL = (0.5, 45.0)
BAND_DELTA_FREE = (4.0, 40.0)
EPOCH_DURATION = 10.0

# Subject classification (LOCKED from TO-CS-02 Appendix B)
SUBJECT_CLASSIFICATION = {
    "02": "drowsy",   "03": "drowsy",   "05": "drowsy",
    "06": "drowsy",   "08": "drowsy",   "10": "drowsy",
    "18": "drowsy",
    "07": "responsive", "09": "responsive", "13": "responsive",
    "14": "responsive", "20": "responsive", "22": "responsive",
    "23": "responsive", "24": "responsive", "25": "responsive",
    "26": "responsive", "27": "responsive", "28": "responsive",
    "29": "responsive",
}

CONDITION_ORDER = ["baseline", "mild", "moderate", "recovery"]

# ROI definitions (same as run_pipeline.py)
ROI_CHANNELS = {
    "F_L": ["Fp1", "F3", "F7", "E20", "E23", "E26", "E27", "E28", "E29", "E30"],
    "F_R": ["Fp2", "Fz", "F4", "F8", "E2", "E3", "E4", "E5", "E6", "E7", "E10", "E123", "E118"],
    "C_L": ["C3", "E31", "E34", "E35", "E37", "E39", "E40", "E41", "E42", "T3"],
    "C_R": ["C4", "Cz", "T4", "E105", "E106", "E109", "E110", "E111", "E112", "E115", "E116"],
    "P_L": ["P3", "E46", "E47", "E50", "E51", "E53", "E54", "E55", "T5", "E59", "E60"],
    "P_R": ["P4", "Pz", "T6", "E85", "E86", "E87", "E90", "E91", "E93", "E97", "E98", "E117"],
    "O_L": ["O1", "E61", "E65", "E66", "E67", "E71", "E72", "E76", "E77", "E78"],
    "O_R": ["O2", "Oz", "E79", "E80", "E84", "E101", "E102", "E103", "E15", "E16", "E18", "E19", "E13"],
}
ROI_NAMES = list(ROI_CHANNELS.keys())


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ============================================================
# DATA LOADING — reuses exact same logic as run_pipeline.py
# ============================================================

def find_subject_files(data_dir):
    """Find and group .set files by subject, ordered by condition."""
    subjects = defaultdict(list)
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.set'):
            continue
        parts = fname.split('-')
        if len(parts) >= 2:
            sub_id = parts[0]
            subjects[sub_id].append(fname)
    for sub_id in subjects:
        subjects[sub_id] = sorted(subjects[sub_id])
    return dict(subjects)


def _read_hdf5_set(filepath):
    """Read EEGLAB .set file in MATLAB v7.3 (HDF5) format."""
    with h5py.File(filepath, 'r') as f:
        eeg = f['EEG']
        sfreq = float(np.array(eeg['srate']).flat[0])
        nbchan = int(np.array(eeg['nbchan']).flat[0])
        pnts = int(np.array(eeg['pnts']).flat[0])

        chanlocs = eeg['chanlocs']
        ch_names = []
        if 'labels' in chanlocs:
            labels_ref = chanlocs['labels']
            for i in range(nbchan):
                try:
                    ref = labels_ref[i, 0]
                    chars = f[ref][()]
                    name = ''.join(chr(c) for c in chars.flat)
                    ch_names.append(name.strip())
                except Exception:
                    ch_names.append(f'E{i+1}')
        else:
            ch_names = [f'E{i+1}' for i in range(nbchan)]

        # Skip uint16 filename refs in data field
        data = None
        if 'data' in eeg:
            d = eeg['data']
            if (isinstance(d, h5py.Dataset) and
                    d.dtype in (np.float32, np.float64) and
                    d.size == nbchan * pnts):
                data = np.array(d, dtype=np.float64)
                if data.shape[0] == pnts and data.shape[1] == nbchan:
                    data = data.T

    if data is None:
        fdt_path = filepath.replace('.set', '.fdt')
        if os.path.exists(fdt_path):
            raw_data = np.fromfile(fdt_path, dtype=np.float32)
            n_expected = nbchan * pnts
            if raw_data.size == n_expected:
                data = raw_data.reshape((pnts, nbchan)).T
            elif raw_data.size % nbchan == 0:
                actual_pnts = raw_data.size // nbchan
                data = raw_data.reshape((actual_pnts, nbchan)).T
                pnts = actual_pnts
            else:
                raise ValueError(f"Cannot reshape .fdt: {raw_data.size} samples, {nbchan} ch")
            data = data.astype(np.float64)
        else:
            raise FileNotFoundError(f"No .fdt file: {fdt_path}")

    return ch_names, sfreq, data


def load_eeg(filepath):
    """Load EEGLAB .set file, return (raw, epochs)."""
    try:
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
    except Exception as e:
        if 'v7.3' in str(e) or 'HDF' in str(e) or 'h5py' in str(e).lower():
            ch_names, sfreq, data = _read_hdf5_set(filepath)
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            raw = mne.io.RawArray(data * 1e-6, info, verbose=False)
        else:
            raise
    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_DURATION, preload=True, overlap=0, verbose=False
    )
    return raw, epochs


def map_channels_to_rois(ch_names):
    """Map channel names to ROI indices."""
    roi_indices = {}
    for roi_name, roi_channels in ROI_CHANNELS.items():
        indices = [ch_names.index(ch) for ch in roi_channels if ch in ch_names]
        roi_indices[roi_name] = indices
    return roi_indices


def get_roi_signals(epochs_data, roi_indices):
    """Average epoch data within each ROI. Returns (n_epochs, n_rois, n_times)."""
    n_epochs, n_channels, n_times = epochs_data.shape
    n_rois = len(ROI_NAMES)
    roi_data = np.zeros((n_epochs, n_rois, n_times))
    for i, roi_name in enumerate(ROI_NAMES):
        idx = roi_indices[roi_name]
        if len(idx) > 0:
            roi_data[:, i, :] = epochs_data[:, idx, :].mean(axis=1)
        else:
            roi_data[:, i, :] = np.nan
    return roi_data


# ============================================================
# OMEGA ESTIMATORS — the three variants
# ============================================================

def compute_psd(signal, sfreq, band):
    """Compute PSD in a frequency band using Welch. Returns (freqs, psd) in band."""
    nperseg = min(SE_NPERSEG, len(signal))
    f, psd = welch(signal, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (f >= band[0]) & (f <= band[1])
    return f[mask], psd[mask]


def se_from_psd(psd_band):
    """Compute normalised spectral entropy from a PSD array."""
    if len(psd_band) == 0 or psd_band.sum() == 0:
        return np.nan
    psd_norm = psd_band / psd_band.sum()
    psd_norm = psd_norm[psd_norm > 0]
    H = -np.sum(psd_norm * np.log2(psd_norm))
    H_max = np.log2(len(psd_norm))
    return H / H_max if H_max > 0 else 0.0


def omega_A(roi_epochs, sfreq, band):
    """
    Omega_A = SE(mean(PSD_epoch)) — "preview-like"
    Compute PSD per epoch, average PSDs, then SE on the mean PSD.
    """
    n_epochs = roi_epochs.shape[0]
    all_psds = []
    for ep in range(n_epochs):
        _, psd = compute_psd(roi_epochs[ep], sfreq, band)
        all_psds.append(psd)
    mean_psd = np.mean(all_psds, axis=0)
    return se_from_psd(mean_psd)


def omega_B(roi_epochs, sfreq, band):
    """
    Omega_B = mean(SE(PSD_epoch)) — "pipeline-like"
    Compute SE per epoch, then average SE values.
    """
    n_epochs = roi_epochs.shape[0]
    se_values = []
    for ep in range(n_epochs):
        _, psd = compute_psd(roi_epochs[ep], sfreq, band)
        se_values.append(se_from_psd(psd))
    return np.nanmean(se_values)


def omega_C(roi_epochs, sfreq, band):
    """
    Omega_C = SE(median(PSD_epoch)) — "robust control"
    Compute PSD per epoch, take median PSD, then SE.
    """
    n_epochs = roi_epochs.shape[0]
    all_psds = []
    for ep in range(n_epochs):
        _, psd = compute_psd(roi_epochs[ep], sfreq, band)
        all_psds.append(psd)
    median_psd = np.median(all_psds, axis=0)
    return se_from_psd(median_psd)


# ============================================================
# STATISTICS
# ============================================================

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    sp = np.sqrt(((n1-1)*np.var(g1, ddof=1) + (n2-1)*np.var(g2, ddof=1)) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / sp if sp > 0 else 0.0


# ============================================================
# MAIN
# ============================================================

def run_parity_test(data_dir):
    log("=" * 70)
    log("OMEGA PARITY TEST: Estimator Order Sensitivity")
    log("SE(mean(PSD)) vs mean(SE(PSD)) vs SE(median(PSD))")
    log("=" * 70)

    subject_files = find_subject_files(data_dir)
    log(f"Found {len(subject_files)} subjects")

    results = {}  # sub_id -> condition -> {variant -> value}

    for sub_id, files in sorted(subject_files.items()):
        if sub_id not in SUBJECT_CLASSIFICATION:
            continue
        group = SUBJECT_CLASSIFICATION[sub_id]
        log(f"\n  Sub-{sub_id} ({group}): {len(files)} conditions")

        results[sub_id] = {}

        for cond_idx, fname in enumerate(files):
            if cond_idx >= len(CONDITION_ORDER):
                break
            condition = CONDITION_ORDER[cond_idx]
            filepath = os.path.join(data_dir, fname)

            # Skip LFS pointers
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    if f.readline().startswith('version https://git-lfs'):
                        continue
            except UnicodeDecodeError:
                pass

            try:
                raw, epochs = load_eeg(filepath)
                sfreq = raw.info['sfreq']
                roi_indices = map_channels_to_rois(raw.ch_names)
                roi_data = get_roi_signals(epochs.get_data(), roi_indices)
                n_epochs = roi_data.shape[0]

                # Compute three Omega variants, averaged across ROIs
                variants = {}
                for band_name, band in [("full", BAND_FULL), ("4-40", BAND_DELTA_FREE)]:
                    a_vals, b_vals, c_vals = [], [], []
                    for r in range(len(ROI_NAMES)):
                        ep_data = roi_data[:, r, :]  # (n_epochs, n_times)
                        if np.all(np.isnan(ep_data)):
                            continue
                        a_vals.append(omega_A(ep_data, sfreq, band))
                        b_vals.append(omega_B(ep_data, sfreq, band))
                        c_vals.append(omega_C(ep_data, sfreq, band))

                    variants[f"A_{band_name}"] = np.nanmean(a_vals)
                    variants[f"B_{band_name}"] = np.nanmean(b_vals)
                    variants[f"C_{band_name}"] = np.nanmean(c_vals)

                results[sub_id][condition] = variants

                if condition == 'moderate':
                    log(f"    {condition} ({n_epochs} ep): "
                        f"A_full={variants['A_full']:.4f}  "
                        f"B_full={variants['B_full']:.4f}  "
                        f"C_full={variants['C_full']:.4f}  "
                        f"delta(A-B)={variants['A_full']-variants['B_full']:.4f}")
                else:
                    log(f"    {condition}: OK ({n_epochs} ep)")

            except Exception as e:
                log(f"    {condition}: ERROR - {e}")

    # ============================================================
    # ANALYSIS: Cohen's d at moderate
    # ============================================================
    log("\n" + "=" * 70)
    log("RESULTS: Cohen's d (responsive vs drowsy at MODERATE)")
    log("=" * 70)

    variant_names = ["A_full", "B_full", "C_full", "A_4-40", "B_4-40", "C_4-40"]
    resp_vals = {v: [] for v in variant_names}
    drow_vals = {v: [] for v in variant_names}

    for sub_id, conditions in results.items():
        if 'moderate' not in conditions:
            continue
        group = SUBJECT_CLASSIFICATION.get(sub_id)
        if not group:
            continue
        mod = conditions['moderate']
        target = resp_vals if group == 'responsive' else drow_vals
        for v in variant_names:
            if v in mod and not np.isnan(mod[v]):
                target[v].append(mod[v])

    log(f"\n  Responsive N = {len(resp_vals['A_full'])}")
    log(f"  Drowsy N = {len(drow_vals['A_full'])}")

    log(f"\n  {'Variant':<12} {'d':>8} {'Resp mean':>10} {'Drow mean':>10}")
    log(f"  {'-'*42}")

    d_values = {}
    for v in variant_names:
        r = np.array(resp_vals[v])
        d = np.array(drow_vals[v])
        if len(r) > 1 and len(d) > 1:
            d_val = cohens_d(r, d)
            d_values[v] = d_val
            log(f"  {v:<12} {d_val:>8.3f} {np.mean(r):>10.4f} {np.mean(d):>10.4f}")

    # ============================================================
    # KEY COMPARISON
    # ============================================================
    log(f"\n{'=' * 70}")
    log("KEY COMPARISON: Does estimator order explain the collapse?")
    log(f"{'=' * 70}")

    if "A_full" in d_values and "B_full" in d_values:
        dA = d_values["A_full"]
        dB = d_values["B_full"]
        log(f"\n  d(Omega_A, full) = {dA:.3f}   <-- SE(mean(PSD)) 'preview method'")
        log(f"  d(Omega_B, full) = {dB:.3f}   <-- mean(SE(PSD)) 'pipeline method'")
        if dB != 0:
            log(f"  Ratio: d_A / d_B = {dA/dB:.1f}x")
        log(f"\n  Preview reported: d = 1.54")
        log(f"  Pipeline reported: d = 0.23")

        if abs(dA) > 1.0 and abs(dB) < 0.5:
            log(f"\n  CONFIRMED: Estimator order explains the collapse.")
            log(f"  SE(mean(PSD)) preserves group contrast; mean(SE(PSD)) destroys it.")
            log(f"  This is Jensen's inequality on spectral entropy.")
        elif abs(dA - dB) < 0.3:
            log(f"\n  NOT CONFIRMED: Both estimators give similar d.")
            log(f"  The collapse has a different cause.")
        else:
            log(f"\n  PARTIAL: Some estimator effect but not the full story.")

    if "A_4-40" in d_values and "B_4-40" in d_values:
        log(f"\n  Delta-free band:")
        log(f"  d(Omega_A, 4-40) = {d_values['A_4-40']:.3f}")
        log(f"  d(Omega_B, 4-40) = {d_values['B_4-40']:.3f}")

    # ============================================================
    # PER-SUBJECT TABLE
    # ============================================================
    log(f"\n{'=' * 70}")
    log("PER-SUBJECT VALUES AT MODERATE (full band)")
    log(f"{'=' * 70}")
    log(f"  {'Subject':<8} {'Group':<12} {'A':>8} {'B':>8} {'C':>8} {'A-B':>8}")
    log(f"  {'-'*50}")

    for sub_id in sorted(results.keys()):
        if 'moderate' not in results[sub_id]:
            continue
        m = results[sub_id]['moderate']
        group = SUBJECT_CLASSIFICATION.get(sub_id, '?')
        oA = m.get('A_full', np.nan)
        oB = m.get('B_full', np.nan)
        oC = m.get('C_full', np.nan)
        log(f"  {sub_id:<8} {group:<12} {oA:>8.4f} {oB:>8.4f} {oC:>8.4f} {oA-oB:>8.4f}")

    # ============================================================
    # CONDITION TRAJECTORY
    # ============================================================
    log(f"\n{'=' * 70}")
    log("CONDITION TRAJECTORY: d(Omega_A) vs d(Omega_B) across all conditions")
    log(f"{'=' * 70}")

    for cond in CONDITION_ORDER:
        rA, rB, dA_list, dB_list = [], [], [], []
        for sub_id, conditions in results.items():
            if cond not in conditions:
                continue
            group = SUBJECT_CLASSIFICATION.get(sub_id)
            if not group:
                continue
            m = conditions[cond]
            if group == 'responsive':
                rA.append(m.get('A_full', np.nan))
                rB.append(m.get('B_full', np.nan))
            else:
                dA_list.append(m.get('A_full', np.nan))
                dB_list.append(m.get('B_full', np.nan))

        rA, rB = np.array(rA), np.array(rB)
        dA_arr, dB_arr = np.array(dA_list), np.array(dB_list)

        if len(rA) > 1 and len(dA_arr) > 1:
            d_a = cohens_d(rA[~np.isnan(rA)], dA_arr[~np.isnan(dA_arr)])
            d_b = cohens_d(rB[~np.isnan(rB)], dB_arr[~np.isnan(dB_arr)])
            ratio_str = f"ratio={d_a/d_b:.1f}x" if d_b != 0 else ""
            log(f"  {cond:<12}  d(A)={d_a:>7.3f}  d(B)={d_b:>7.3f}  {ratio_str}")

    # Save
    output = {
        "test": "omega_parity",
        "date": datetime.now().isoformat(),
        "d_values": {k: float(v) for k, v in d_values.items()},
        "n_responsive": len(resp_vals.get('A_full', [])),
        "n_drowsy": len(drow_vals.get('A_full', [])),
    }
    outpath = os.path.join(os.path.dirname(data_dir) or '.', "omega_parity_results.json")
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    log(f"\nResults saved: {outpath}")

    log(f"\n{'=' * 70}")
    log("PARITY TEST COMPLETE")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    run_parity_test(args.data_dir)
