#!/usr/bin/env python3
"""
TO-CS-02: Complete EFC-C Product Hypothesis Pipeline
Tests C = Omega x Kappa on Chennu et al. 2016 propofol sedation dataset.

All parameters locked per Appendix B (pre-registered before seeing TE data):
- ROI mapping: 8 ROIs from 90 channels (coordinate-based quantile split)
- Omega: Spectral Entropy (Welch PSD, normalized) at full (0.5-45 Hz) and delta-free (4-40 Hz)
- Kappa: Transfer Entropy (pyinform, k=3, 6 bins) on 56 directed ROI pairs
- Statistics: Bootstrap delta-d, N=10000, seed=42, Bonferroni alpha=0.0125
- Decision tree: F1/C1/C2/C3 scoring

Usage:
    pip install mne numpy scipy pyinform h5py
    python run_pipeline.py --data-dir "path/to/Sedation-RestingState"
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
from collections import defaultdict

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import mne
    mne.set_log_level('ERROR')
except ImportError:
    print("ERROR: mne not installed. Run: pip install mne")
    sys.exit(1)

try:
    from scipy import signal, stats
    from scipy.io import loadmat
except ImportError:
    print("ERROR: scipy not installed. Run: pip install scipy")
    sys.exit(1)

try:
    import pyinform
    from pyinform.transferentropy import transfer_entropy
except ImportError:
    print("ERROR: pyinform not installed. Run: pip install pyinform")
    sys.exit(1)

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ============================================================
# LOCKED PARAMETERS (Appendix B)
# ============================================================

# ROI definitions: EGI HydroCel GSN-128 -> 8 ROIs
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

# Subject classification (locked: <24/40 correct at moderate = drowsy)
# Subject numbering in filenames vs preview doc mapping
SUBJECT_CLASSIFICATION = {
    # Drowsy (N=7): Sub-02,04,05,06,08,10,17 in preview
    # File numbering: 02,03,05,06,08,10,18 (offset by file naming)
    "02": "drowsy",   # Sub-02: 3/40
    "03": "drowsy",   # Sub-04: 1/40  (file 03 = preview Sub-04)
    "05": "drowsy",   # Sub-05: 6/40
    "06": "drowsy",   # Sub-06: 21/40
    "08": "drowsy",   # Sub-08: 0/40
    "10": "drowsy",   # Sub-10: 12/40
    "18": "drowsy",   # Sub-17: 15/40 (file 18 = preview Sub-17)
    # Responsive (N=13): all others
    "07": "responsive",
    "09": "responsive",
    "13": "responsive",
    "14": "responsive",
    "20": "responsive",
    "22": "responsive",
    "23": "responsive",
    "24": "responsive",
    "25": "responsive",
    "26": "responsive",
    "27": "responsive",
    "28": "responsive",
    "29": "responsive",
}

# Condition mapping: file suffix -> condition name
# Files have 4 conditions per subject, ordered by recording number
CONDITION_ORDER = ["baseline", "mild", "moderate", "recovery"]

# Entropy parameters
PE_ORDER = 3
PE_DELAY = 1
SE_NPERSEG = 512  # Welch segment length for spectral entropy
FREQ_FULL = (0.5, 45.0)
FREQ_DELTA_FREE = (4.0, 40.0)

# Transfer entropy parameters
TE_K = 3          # History length
TE_BINS = 6       # Discretization bins

# Bootstrap parameters
N_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42
ALPHA = 0.0125    # Bonferroni: 0.05/4

# Epoch parameters
EPOCH_DURATION = 10.0  # seconds


# ============================================================
# DATA LOADING
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

    # Sort each subject's files (natural order = condition order)
    for sub_id in subjects:
        subjects[sub_id] = sorted(subjects[sub_id])

    return dict(subjects)


def _read_hdf5_set(filepath):
    """
    Read EEGLAB .set file saved in MATLAB v7.3 (HDF5) format using h5py.
    Returns channel names, sampling rate, and data array (n_channels, n_times).
    The .fdt file contains the raw float data referenced by the .set file.
    """
    if not HAS_H5PY:
        raise ImportError("h5py is required for MATLAB v7.3 .set files. "
                          "Run: pip install h5py")

    with h5py.File(filepath, 'r') as f:
        eeg = f['EEG']

        # Sampling rate
        sfreq = float(np.array(eeg['srate']).flat[0])

        # Number of channels and points
        nbchan = int(np.array(eeg['nbchan']).flat[0])
        pnts = int(np.array(eeg['pnts']).flat[0])

        # Channel names - stored as HDF5 references in v7.3 format
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

        # Debug info for first file
        print(f"      [HDF5] nbchan={nbchan}, pnts={pnts}, sfreq={sfreq}")
        print(f"      [HDF5] ch_names[0:5]={ch_names[:5]}")
        if 'data' in eeg:
            d_info = eeg['data']
            if isinstance(d_info, h5py.Dataset):
                print(f"      [HDF5] data in .set: shape={d_info.shape}, dtype={d_info.dtype}")
            else:
                print(f"      [HDF5] data in .set: type={type(d_info)}")

        # Try to read data from the .set file directly
        data = None
        if 'data' in eeg:
            d = eeg['data']
            # Only use in-file data if it's a float dataset with the right size
            # (uint16 with small shape = filename string reference to .fdt)
            if (isinstance(d, h5py.Dataset) and
                    d.dtype in (np.float32, np.float64) and
                    d.size == nbchan * pnts):
                data = np.array(d, dtype=np.float64)
                if data.shape[0] == pnts and data.shape[1] == nbchan:
                    data = data.T  # Transpose to (channels, timepoints)

    # If data not in .set, read from .fdt file
    if data is None:
        fdt_path = filepath.replace('.set', '.fdt')
        if os.path.exists(fdt_path):
            raw_data = np.fromfile(fdt_path, dtype=np.float32)
            n_samples_expected = nbchan * pnts

            if raw_data.size == n_samples_expected:
                # EEGLAB .fdt: stored as interleaved channels (C-order)
                # Layout: [ch0_t0, ch1_t0, ..., chN_t0, ch0_t1, ch1_t1, ...]
                data = raw_data.reshape((pnts, nbchan)).T
            elif raw_data.size % nbchan == 0:
                # File size doesn't match expected pnts, recalculate
                actual_pnts = raw_data.size // nbchan
                data = raw_data.reshape((actual_pnts, nbchan)).T
                pnts = actual_pnts
            else:
                raise ValueError(
                    f"Cannot reshape .fdt data: {raw_data.size} samples, "
                    f"{nbchan} channels, expected {n_samples_expected}"
                )
            data = data.astype(np.float64)
        else:
            raise FileNotFoundError(f"Cannot find .fdt file: {fdt_path}")

    return ch_names, sfreq, data


def load_eeg(filepath, epoch_duration=EPOCH_DURATION):
    """Load EEGLAB .set file and create fixed-length epochs.
    Handles both standard and MATLAB v7.3 (HDF5) formats."""
    try:
        # Try standard MNE EEGLAB reader first
        raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
    except Exception as e:
        if 'v7.3' in str(e) or 'HDF' in str(e) or 'h5py' in str(e).lower():
            # MATLAB v7.3 format - use h5py reader
            ch_names, sfreq, data = _read_hdf5_set(filepath)

            # Create MNE RawArray
            info = mne.create_info(
                ch_names=ch_names,
                sfreq=sfreq,
                ch_types='eeg'
            )
            raw = mne.io.RawArray(data * 1e-6, info, verbose=False)  # uV -> V
        else:
            raise

    # Create fixed-length epochs
    epochs = mne.make_fixed_length_epochs(
        raw, duration=epoch_duration, preload=True, overlap=0, verbose=False
    )

    return raw, epochs


def map_channels_to_rois(raw):
    """Map available channels to ROIs, return indices."""
    available = raw.ch_names
    roi_indices = {}

    for roi_name, roi_channels in ROI_CHANNELS.items():
        indices = []
        for ch in roi_channels:
            if ch in available:
                indices.append(available.index(ch))
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
# OMEGA (Differentiation)
# ============================================================

def permutation_entropy(x, order=PE_ORDER, delay=PE_DELAY):
    """Compute normalized permutation entropy of a 1D signal."""
    n = len(x)
    import math
    n_perms = math.factorial(order)

    # Build permutation patterns
    indices = np.arange(order) * delay
    patterns = []
    for i in range(n - (order - 1) * delay):
        segment = x[i + indices]
        pattern = tuple(np.argsort(segment))
        patterns.append(pattern)

    if len(patterns) == 0:
        return np.nan

    # Count pattern frequencies
    from collections import Counter
    counts = Counter(patterns)
    probs = np.array(list(counts.values()), dtype=float) / len(patterns)

    # Shannon entropy, normalized
    H = -np.sum(probs * np.log2(probs))
    H_max = np.log2(n_perms)

    return H / H_max if H_max > 0 else 0.0


def spectral_entropy(x, sfreq, freq_range, nperseg=SE_NPERSEG):
    """Compute normalized spectral entropy using Welch PSD."""
    f, psd = signal.welch(x, fs=sfreq, nperseg=min(nperseg, len(x)),
                          noverlap=min(nperseg // 2, len(x) // 2))

    # Select frequency range
    mask = (f >= freq_range[0]) & (f <= freq_range[1])
    psd_band = psd[mask]

    if len(psd_band) == 0 or np.sum(psd_band) == 0:
        return np.nan

    # Normalize to probability distribution
    psd_norm = psd_band / np.sum(psd_band)
    psd_norm = psd_norm[psd_norm > 0]

    # Shannon entropy, normalized
    H = -np.sum(psd_norm * np.log2(psd_norm))
    H_max = np.log2(len(psd_band))

    return H / H_max if H_max > 0 else 0.0


def compute_omega(roi_data, sfreq):
    """
    Compute Omega for each ROI and epoch.
    Returns dict with PE, SE_full, SE_delta_free for each ROI.
    """
    n_epochs, n_rois, n_times = roi_data.shape

    omega = {
        'PE': np.zeros((n_epochs, n_rois)),
        'SE_full': np.zeros((n_epochs, n_rois)),
        'SE_4_40': np.zeros((n_epochs, n_rois)),
    }

    for ep in range(n_epochs):
        for r in range(n_rois):
            x = roi_data[ep, r, :]
            if np.all(np.isnan(x)):
                omega['PE'][ep, r] = np.nan
                omega['SE_full'][ep, r] = np.nan
                omega['SE_4_40'][ep, r] = np.nan
                continue

            omega['PE'][ep, r] = permutation_entropy(x)
            omega['SE_full'][ep, r] = spectral_entropy(x, sfreq, FREQ_FULL)
            omega['SE_4_40'][ep, r] = spectral_entropy(x, sfreq, FREQ_DELTA_FREE)

    return omega


# ============================================================
# KAPPA (Integration / Transfer Entropy)
# ============================================================

def discretize(x, n_bins=TE_BINS):
    """Discretize continuous signal into n_bins equal-width bins."""
    x_min, x_max = np.min(x), np.max(x)
    if x_max == x_min:
        return np.zeros(len(x), dtype=int)
    bins = np.linspace(x_min, x_max, n_bins + 1)
    digitized = np.digitize(x, bins[1:-1])
    return digitized


def compute_kappa(roi_data):
    """
    Compute Transfer Entropy between all 56 directed ROI pairs.
    Returns (n_epochs, 8, 8) matrix where [i,j] = TE from ROI_i -> ROI_j.
    Diagonal is NaN.
    """
    n_epochs, n_rois, n_times = roi_data.shape
    te_matrix = np.full((n_epochs, n_rois, n_rois), np.nan)

    for ep in range(n_epochs):
        for i in range(n_rois):
            for j in range(n_rois):
                if i == j:
                    continue

                source = roi_data[ep, i, :]
                target = roi_data[ep, j, :]

                if np.all(np.isnan(source)) or np.all(np.isnan(target)):
                    continue

                # Discretize
                src_d = discretize(source)
                tgt_d = discretize(target)

                try:
                    te = transfer_entropy(src_d, tgt_d, k=TE_K)
                    te_matrix[ep, i, j] = te
                except Exception:
                    te_matrix[ep, i, j] = np.nan

    return te_matrix


def extract_kappa_metrics(te_matrix):
    """
    Extract key kappa metrics from TE matrix.
    te_matrix: (n_epochs, 8, 8)
    """
    n_epochs = te_matrix.shape[0]
    roi_idx = {name: i for i, name in enumerate(ROI_NAMES)}

    metrics = {}

    # Global kappa: mean of all 56 pairs
    mask = ~np.eye(8, dtype=bool)
    metrics['kappa_global'] = np.array([
        np.nanmean(te_matrix[ep][mask]) for ep in range(n_epochs)
    ])

    # Hub kappa for P_L
    pl = roi_idx['P_L']
    others = [i for i in range(8) if i != pl]

    metrics['kappa_in_PL'] = np.array([
        np.nanmean([te_matrix[ep, o, pl] for o in others]) for ep in range(n_epochs)
    ])
    metrics['kappa_out_PL'] = np.array([
        np.nanmean([te_matrix[ep, pl, o] for o in others]) for ep in range(n_epochs)
    ])
    metrics['kappa_hub_PL'] = (metrics['kappa_in_PL'] + metrics['kappa_out_PL']) / 2

    # Hub kappa for C_R
    cr = roi_idx['C_R']
    others_cr = [i for i in range(8) if i != cr]

    metrics['kappa_hub_CR'] = np.array([
        np.nanmean([te_matrix[ep, o, cr] for o in others_cr] +
                    [te_matrix[ep, cr, o] for o in others_cr])
        for ep in range(n_epochs)
    ])

    # Hub kappa for O_L (negative control)
    ol = roi_idx['O_L']
    others_ol = [i for i in range(8) if i != ol]

    metrics['kappa_hub_OL'] = np.array([
        np.nanmean([te_matrix[ep, o, ol] for o in others_ol] +
                    [te_matrix[ep, ol, o] for o in others_ol])
        for ep in range(n_epochs)
    ])

    # Directional: F -> P_L and P_L -> F
    fl = roi_idx['F_L']
    fr = roi_idx['F_R']

    metrics['TE_F_to_PL'] = np.array([
        np.nanmean([te_matrix[ep, fl, pl], te_matrix[ep, fr, pl]])
        for ep in range(n_epochs)
    ])
    metrics['TE_PL_to_F'] = np.array([
        np.nanmean([te_matrix[ep, pl, fl], te_matrix[ep, pl, fr]])
        for ep in range(n_epochs)
    ])

    return metrics


# ============================================================
# C PRODUCT & STATISTICS
# ============================================================

def compute_c_products(omega, kappa_metrics):
    """Compute C = Omega x Kappa products (per Appendix B)."""
    products = {}

    # Mean Omega across ROIs (per epoch)
    omega_global_full = np.nanmean(omega['SE_full'], axis=1)
    omega_global_4_40 = np.nanmean(omega['SE_4_40'], axis=1)

    # P_L specific Omega
    pl_idx = ROI_NAMES.index('P_L')
    omega_PL_full = omega['SE_full'][:, pl_idx]
    omega_PL_4_40 = omega['SE_4_40'][:, pl_idx]

    # O_L specific Omega (negative control)
    ol_idx = ROI_NAMES.index('O_L')
    omega_OL_full = omega['SE_full'][:, ol_idx]

    # Global C
    products['C_global_full'] = omega_global_full * kappa_metrics['kappa_global']
    products['C_global_4_40'] = omega_global_4_40 * kappa_metrics['kappa_global']

    # P_L C
    products['C_PL'] = omega_PL_4_40 * kappa_metrics['kappa_hub_PL']

    # C_R C
    cr_idx = ROI_NAMES.index('C_R')
    omega_CR = omega['SE_4_40'][:, cr_idx]
    products['C_CR'] = omega_CR * kappa_metrics['kappa_hub_CR']

    # Occipital (negative control)
    products['C_occipital'] = omega_OL_full * kappa_metrics['kappa_hub_OL']

    # Directional
    products['C_topdown'] = omega_PL_4_40 * kappa_metrics['TE_F_to_PL']
    products['C_bottomup'] = omega_PL_4_40 * kappa_metrics['TE_PL_to_F']
    products['A_FP'] = products['C_topdown'] - products['C_bottomup']

    # Additive control
    products['C_add_global_4_40'] = omega_global_4_40 + kappa_metrics['kappa_global']
    products['C_add_global_full'] = omega_global_full + kappa_metrics['kappa_global']

    # Store Omega and Kappa separately for comparison
    products['Omega_global_full'] = omega_global_full
    products['Omega_global_4_40'] = omega_global_4_40
    products['Omega_PL_4_40'] = omega_PL_4_40
    products['Kappa_global'] = kappa_metrics['kappa_global']
    products['Kappa_hub_PL'] = kappa_metrics['kappa_hub_PL']

    return products


def cohens_d(group1, group2):
    """Compute Cohen's d (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_sd = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) +
                         (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return np.nan
    return mean_diff / pooled_sd


def bootstrap_delta_d(responsive_A, drowsy_A, responsive_B, drowsy_B,
                      n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """
    Bootstrap test: does measure A beat measure B?
    Returns delta_d, CI_lower, CI_upper, p_value.
    """
    rng = np.random.RandomState(seed)
    n_resp = len(responsive_A)
    n_drow = len(drowsy_A)

    delta_ds = []
    for _ in range(n_boot):
        # Resample within groups (paired indices)
        idx_resp = rng.choice(n_resp, size=n_resp, replace=True)
        idx_drow = rng.choice(n_drow, size=n_drow, replace=True)

        d_A = cohens_d(responsive_A[idx_resp], drowsy_A[idx_drow])
        d_B = cohens_d(responsive_B[idx_resp], drowsy_B[idx_drow])
        delta_ds.append(d_A - d_B)

    delta_ds = np.array(delta_ds)
    observed_delta = cohens_d(responsive_A, drowsy_A) - cohens_d(responsive_B, drowsy_B)

    ci_lower = np.percentile(delta_ds, 2.5)
    ci_upper = np.percentile(delta_ds, 97.5)
    p_value = np.mean(delta_ds <= 0)  # One-tailed

    return observed_delta, ci_lower, ci_upper, p_value


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(data_dir):
    """Run the complete TO-CS-02 pipeline."""
    print("=" * 70)
    print("TO-CS-02: EFC-C Product Hypothesis Test")
    print("C = Omega x Kappa on Chennu et al. 2016 Propofol Dataset")
    print("=" * 70)

    # Phase 1: Load data
    print("\n[Phase 1] Loading EEG data...")
    subject_files = find_subject_files(data_dir)

    all_results = {}  # sub_id -> condition -> {omega, kappa_metrics, products}

    for sub_id, files in sorted(subject_files.items()):
        if sub_id not in SUBJECT_CLASSIFICATION:
            print(f"  Sub-{sub_id}: SKIPPED (not in classification)")
            continue

        group = SUBJECT_CLASSIFICATION[sub_id]
        print(f"\n  Sub-{sub_id} ({group}): {len(files)} conditions")
        all_results[sub_id] = {}

        for cond_idx, fname in enumerate(files):
            if cond_idx >= len(CONDITION_ORDER):
                break
            condition = CONDITION_ORDER[cond_idx]

            filepath = os.path.join(data_dir, fname)

            # Check for LFS pointer
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                if first_line.startswith('version https://git-lfs.github.com/spec/v1'):
                    print(f"    {condition}: SKIPPED (LFS pointer)")
                    continue
            except UnicodeDecodeError:
                pass  # Binary = real data

            try:
                raw, epochs = load_eeg(filepath)
                sfreq = raw.info['sfreq']
                roi_indices = map_channels_to_rois(raw)

                n_mapped = sum(len(v) for v in roi_indices.values())
                print(f"    {condition}: {len(epochs)} epochs, "
                      f"{len(raw.ch_names)} ch -> {n_mapped} mapped to ROIs")

                # Get ROI signals
                roi_data = get_roi_signals(epochs.get_data(), roi_indices)

                # Phase 2: Compute Omega
                omega = compute_omega(roi_data, sfreq)

                # Phase 3: Compute Kappa
                print(f"      Computing TE (56 pairs x {len(epochs)} epochs)...", end='', flush=True)
                te_matrix = compute_kappa(roi_data)
                kappa_metrics = extract_kappa_metrics(te_matrix)
                print(" done")

                # Compute C products
                products = compute_c_products(omega, kappa_metrics)

                # Store mean over epochs
                result = {}
                for key, values in products.items():
                    result[key] = np.nanmean(values)
                for key in ['PE', 'SE_full', 'SE_4_40']:
                    result[f'Omega_{key}_global'] = np.nanmean(omega[key])

                all_results[sub_id][condition] = result

            except Exception as e:
                print(f"    {condition}: ERROR - {e}")

    # Phase 4: Dissociation check
    print("\n" + "=" * 70)
    print("[Phase 4] Dissociation Check: corr(Omega, Kappa)")
    print("=" * 70)

    omega_vals = []
    kappa_vals = []
    for sub_id, conditions in all_results.items():
        for cond, result in conditions.items():
            if 'Omega_global_4_40' in result and 'Kappa_global' in result:
                omega_vals.append(result['Omega_global_4_40'])
                kappa_vals.append(result['Kappa_global'])

    if len(omega_vals) > 2:
        r, p = stats.pearsonr(omega_vals, kappa_vals)
        print(f"  corr(Omega, Kappa) = {r:.3f} (p={p:.4f}), N={len(omega_vals)}")
        if abs(r) > 0.8:
            print("  STOP RULE TRIGGERED: |r| > 0.8 -> Indeterminate")
            print("  Omega and Kappa are too correlated to test product hypothesis.")
            return
        else:
            print(f"  PASS: |r| = {abs(r):.3f} <= 0.8 -> sufficient dissociation")
    else:
        print("  WARNING: Not enough data points for correlation check")

    # Phase 5: Group comparison at moderate sedation
    print("\n" + "=" * 70)
    print("[Phase 5] Responsive vs Drowsy at Moderate Sedation")
    print("=" * 70)

    # Collect moderate sedation values by group
    responsive = defaultdict(list)
    drowsy = defaultdict(list)

    for sub_id, conditions in all_results.items():
        if 'moderate' not in conditions:
            continue
        group = SUBJECT_CLASSIFICATION.get(sub_id)
        result = conditions['moderate']

        target = responsive if group == 'responsive' else drowsy
        for key, value in result.items():
            if not np.isnan(value):
                target[key].append(value)

    print(f"\n  Responsive N={len(responsive.get('C_global_4_40', []))}, "
          f"Drowsy N={len(drowsy.get('C_global_4_40', []))}")

    # Compute Cohen's d for all measures
    print("\n  Cohen's d (responsive vs drowsy at moderate):")
    print(f"  {'Measure':<30} {'d':>8} {'Resp mean':>10} {'Drow mean':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10}")

    measures_of_interest = [
        'Omega_global_full', 'Omega_global_4_40', 'Kappa_global', 'Kappa_hub_PL',
        'C_global_full', 'C_global_4_40', 'C_PL', 'C_CR', 'C_occipital',
        'C_topdown', 'C_bottomup', 'A_FP',
        'C_add_global_full', 'C_add_global_4_40',
    ]

    d_values = {}
    for measure in measures_of_interest:
        r_vals = np.array(responsive.get(measure, []))
        d_vals = np.array(drowsy.get(measure, []))
        if len(r_vals) >= 2 and len(d_vals) >= 2:
            d = cohens_d(r_vals, d_vals)
            d_values[measure] = d
            print(f"  {measure:<30} {d:>8.3f} {np.mean(r_vals):>10.4f} {np.mean(d_vals):>10.4f}")

    # Phase 5b: Four-way bootstrap comparison
    print("\n" + "=" * 70)
    print("[Phase 5b] Bootstrap Delta-d Comparisons (Appendix B.2)")
    print("=" * 70)

    comparisons = [
        ('a', 'C_global_4_40', 'Omega_global_4_40',
         'C = Omega(4-40) x kappa vs Omega(4-40)'),
        ('b', 'C_global_4_40', 'Kappa_global',
         'C = Omega(4-40) x kappa vs kappa alone'),
        ('c', 'C_global_full', 'Omega_global_full',
         'C = Omega(full) x kappa vs Omega(full)'),
        ('d', 'C_global_4_40', 'C_add_global_4_40',
         'C_mult vs C_add'),
    ]

    comparison_results = {}
    for comp_id, measure_a, measure_b, description in comparisons:
        r_a = np.array(responsive.get(measure_a, []))
        d_a = np.array(drowsy.get(measure_a, []))
        r_b = np.array(responsive.get(measure_b, []))
        d_b = np.array(drowsy.get(measure_b, []))

        if len(r_a) < 2 or len(d_a) < 2 or len(r_b) < 2 or len(d_b) < 2:
            print(f"\n  ({comp_id}) {description}: INSUFFICIENT DATA")
            continue

        delta_d, ci_lo, ci_hi, p = bootstrap_delta_d(r_a, d_a, r_b, d_b)
        passes = delta_d > 0 and ci_lo > 0 and p < ALPHA

        comparison_results[comp_id] = {
            'delta_d': delta_d, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'p': p, 'passes': passes
        }

        status = "PASS" if passes else "FAIL"
        print(f"\n  ({comp_id}) {description}")
        print(f"      Delta-d = {delta_d:.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"      p = {p:.4f}, alpha = {ALPHA}")
        print(f"      Result: {status}")

    # Phase 6: Scoring (Decision Tree)
    print("\n" + "=" * 70)
    print("[Phase 6] Decision Tree Scoring")
    print("=" * 70)

    if 'a' not in comparison_results:
        print("  Cannot score: comparison (a) not computed")
        return

    a = comparison_results.get('a', {})
    c = comparison_results.get('c', {})

    if not a.get('passes', False):
        score = 'F1'
        reason = 'Comparison (a) fails: kappa adds nothing beyond Omega(4-40 Hz)'
    elif not c.get('passes', False):
        score = 'C1'
        reason = 'kappa adds beyond spectral shift only, but not beyond full Omega'
    elif a.get('delta_d', 0) >= 0.5:
        score = 'C3'
        reason = 'Strong support: C beats both Omega measures with delta-d >= 0.5'
    else:
        score = 'C2'
        reason = 'Moderate support: C beats both Omega measures but delta-d < 0.5'

    print(f"\n  SCORE: {score}")
    print(f"  {reason}")

    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  Dissociation: corr(Omega, Kappa) = {r:.3f}")
    print(f"  Omega bar (4-40 Hz):   d = {d_values.get('Omega_global_4_40', float('nan')):.3f} (preview: 0.56)")
    print(f"  Omega bar (full):      d = {d_values.get('Omega_global_full', float('nan')):.3f} (preview: 1.54)")
    print(f"  Kappa global:          d = {d_values.get('Kappa_global', float('nan')):.3f}")
    print(f"  C = Omega(4-40) x K:   d = {d_values.get('C_global_4_40', float('nan')):.3f}")
    print(f"  C = Omega(full) x K:   d = {d_values.get('C_global_full', float('nan')):.3f}")
    print(f"  C_PL (parietal left):  d = {d_values.get('C_PL', float('nan')):.3f}")
    print(f"  C_occipital (control): d = {d_values.get('C_occipital', float('nan')):.3f}")
    print(f"\n  FINAL SCORE: {score}")
    print(f"  {reason}")

    # Save detailed results
    output_path = os.path.join(os.path.dirname(data_dir), 'TO-CS-02_results.json')
    output = {
        'score': score,
        'reason': reason,
        'dissociation_r': float(r),
        'd_values': {k: float(v) for k, v in d_values.items()},
        'comparisons': {
            k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                for kk, vv in v.items()}
            for k, v in comparison_results.items()
        },
        'n_responsive': len(responsive.get('C_global_4_40', [])),
        'n_drowsy': len(drowsy.get('C_global_4_40', [])),
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Detailed results saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TO-CS-02 Pipeline')
    parser.add_argument('--data-dir', required=True,
                        help='Path to Sedation-RestingState directory')
    args = parser.parse_args()

    run_pipeline(args.data_dir)
