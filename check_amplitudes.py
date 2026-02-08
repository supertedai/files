#!/usr/bin/env python3
"""
Check actual peak-to-peak amplitudes per sleep stage.
This tells us what rejection threshold is appropriate.
"""
import mne
import numpy as np
import glob
import os

mne.set_log_level('WARNING')

DATA_DIR = "./anphy_sleep_data"

# Find one subject with N3
edfs = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.edf"), recursive=True))

# Use EPCTL21 (has 192 N3 epochs in scoring)
edf = [e for e in edfs if "EPCTL21" in e][0]
txt = edf.replace(".edf", ".txt")

print(f"Loading {os.path.basename(edf)}...")
raw = mne.io.read_raw_edf(edf, preload=True, verbose=False)

# Strip -Ref suffix
rename = {}
for ch in raw.ch_names:
    if ch.endswith('-Ref'):
        rename[ch] = ch.replace('-Ref', '')
if rename:
    raw.rename_channels(rename)

# Pick EEG channels
exclude = ['EOG', 'EMG', 'ECG', 'EKG', 'ZY', 'SO1', 'SO2', 'RLEG', 'LLEG', 'ChEMG']
eeg_ch = [ch for ch in raw.ch_names if not any(ex.lower() in ch.lower() for ex in exclude)]
raw.pick_channels(eeg_ch)

# Resample
raw.resample(250)
raw.filter(0.5, 45.0, verbose=False)

# Load scoring
scoring = []
with open(txt, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t') if '\t' in line else line.split()
        for p in parts:
            p_clean = p.strip()
            stage_map = {
                'W': 'Wake', 'R': 'REM', 'N1': 'N1', 'N2': 'N2', 'N3': 'N3',
                '0': 'Wake', '1': 'N1', '2': 'N2', '3': 'N3', '4': 'N3', '5': 'REM',
                'WAKE': 'Wake', 'REM': 'REM', 'Wake': 'Wake',
            }
            if p_clean in stage_map:
                scoring.append(stage_map[p_clean])
                break

print(f"Scoring entries: {len(scoring)}")

# Check amplitude per stage
sfreq = raw.info['sfreq']
epoch_len = 30  # standard PSG scoring epoch
n_samples = int(epoch_len * sfreq)
data = raw.get_data() * 1e6  # to uV

print(f"\nAmplitude analysis (peak-to-peak per channel, in uV):")
print(f"{'Stage':<8} {'N_epochs':>10} {'Median PtP':>12} {'P25':>8} {'P75':>8} {'P95':>8} {'Max':>8} {'<100uV':>8} {'<200uV':>8} {'<300uV':>8} {'<500uV':>8}")
print("-" * 110)

for stage in ['Wake', 'N1', 'N2', 'N3', 'REM']:
    ptps = []
    for i, s in enumerate(scoring):
        if s != stage:
            continue
        start = i * n_samples
        end = start + n_samples
        if end > data.shape[1]:
            continue

        epoch = data[:, start:end]
        max_ptp = np.max(np.ptp(epoch, axis=1))  # max across channels
        ptps.append(max_ptp)

    if ptps:
        ptps = np.array(ptps)
        below_100 = np.sum(ptps < 100)
        below_200 = np.sum(ptps < 200)
        below_300 = np.sum(ptps < 300)
        below_500 = np.sum(ptps < 500)
        print(f"{stage:<8} {len(ptps):>10} {np.median(ptps):>12.1f} {np.percentile(ptps,25):>8.1f} "
              f"{np.percentile(ptps,75):>8.1f} {np.percentile(ptps,95):>8.1f} {np.max(ptps):>8.1f} "
              f"{below_100:>8} {below_200:>8} {below_300:>8} {below_500:>8}")

# Also check with 10s sub-epochs within 30s scoring epochs
print(f"\n\nWith 10s sub-epochs (our pipeline setting):")
n_sub = int(10 * sfreq)
print(f"{'Stage':<8} {'N_10s':>10} {'Median PtP':>12} {'<100uV':>8} {'<200uV':>8} {'<300uV':>8} {'<500uV':>8}")
print("-" * 80)

for stage in ['Wake', 'N1', 'N2', 'N3', 'REM']:
    ptps = []
    for i, s in enumerate(scoring):
        if s != stage:
            continue
        base = i * n_samples
        # 3 sub-epochs per 30s epoch
        for sub in range(3):
            start = base + sub * n_sub
            end = start + n_sub
            if end > data.shape[1]:
                continue
            epoch = data[:, start:end]
            max_ptp = np.max(np.ptp(epoch, axis=1))
            ptps.append(max_ptp)

    if ptps:
        ptps = np.array(ptps)
        print(f"{stage:<8} {len(ptps):>10} {np.median(ptps):>12.1f} "
              f"{np.sum(ptps<100):>8} {np.sum(ptps<200):>8} {np.sum(ptps<300):>8} {np.sum(ptps<500):>8}")
