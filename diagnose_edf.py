#!/usr/bin/env python3
"""
Diagnostic: Inspect ANPHY-Sleep EDF files
Run this BEFORE the main pipeline to understand channel names and annotation format.
"""
import mne
import glob
import os
import numpy as np

mne.set_log_level('WARNING')

DATA_DIR = "./anphy_sleep_data"

# Find EDF files
edfs = sorted(glob.glob(os.path.join(DATA_DIR, "**/*.edf"), recursive=True))
print(f"Found {len(edfs)} EDF files\n")

for edf_path in edfs[:2]:  # Just check first 2 subjects
    print("=" * 70)
    print(f"FILE: {os.path.basename(edf_path)}")
    print("=" * 70)

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

    # Channel names
    print(f"\nTotal channels: {len(raw.ch_names)}")
    print(f"Sample rate: {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]/3600:.1f} hours")

    print(f"\nALL CHANNEL NAMES:")
    for i, ch in enumerate(raw.ch_names):
        print(f"  [{i:3d}] '{ch}'")

    # Annotations
    print(f"\nANNOTATIONS ({len(raw.annotations)} total):")
    print(f"{'Onset (s)':>12} {'Duration (s)':>14} {'Description'}")
    print("-" * 50)

    # Show first 20 and last 5
    anns = raw.annotations
    show_indices = list(range(min(20, len(anns))))
    if len(anns) > 25:
        show_indices += list(range(len(anns)-5, len(anns)))

    for idx in show_indices:
        a = anns[idx]
        print(f"  {a['onset']:>10.1f}  {a['duration']:>12.1f}  '{a['description']}'")
        if idx == 19 and len(anns) > 25:
            print(f"  ... ({len(anns) - 25} more) ...")

    # Check: are durations all 0?
    durations = [a['duration'] for a in anns]
    print(f"\nDuration stats: min={min(durations):.1f}, max={max(durations):.1f}, "
          f"mean={np.mean(durations):.1f}")
    if max(durations) == 0:
        print(">>> ALL DURATIONS ARE 0 -- annotations are point events, not intervals!")
        print(">>> Need to infer duration from gap between consecutive annotations")

    # Unique descriptions
    descs = set(a['description'] for a in anns)
    print(f"\nUnique annotation labels: {sorted(descs)}")

    print()
