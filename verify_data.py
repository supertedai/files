#!/usr/bin/env python3
"""
TO-CS-02: Data Verification Script
Checks that EEG files are actual data (not LFS pointers) and validates structure.
Run this FIRST before running the pipeline.

Usage:
    python verify_data.py --data-dir "path/to/Sedation-RestingState"
"""

import os
import sys
import argparse

try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Expected subject IDs from Chennu et al. 2016
EXPECTED_SUBJECTS = [
    "02", "03", "05", "06", "07", "08", "09", "10",
    "13", "14", "18", "20", "22", "23", "24", "25",
    "26", "27", "28", "29"
]

# Subject classification (locked per Appendix B)
DROWSY_SUBJECTS = {"02", "05", "06", "08", "10"}  # Sub IDs with <24/40 at moderate
# Note: Sub-04 and Sub-17 from preview doc map to different file numbering


def check_lfs_pointer(filepath):
    """Check if a file is a Git LFS pointer (not actual data)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
        return first_line.startswith('version https://git-lfs.github.com/spec/v1')
    except (UnicodeDecodeError, OSError):
        return False  # Binary file = actual data


def find_set_files(data_dir):
    """Find all .set files and group by subject."""
    subjects = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.set'):
            continue
        # Parse subject ID from filename: "XX-YYYY-anest YYYYMMDD HHH.NNN.set"
        parts = fname.split('-')
        if len(parts) >= 2:
            sub_id = parts[0]
            if sub_id not in subjects:
                subjects[sub_id] = []
            subjects[sub_id].append(fname)
    return subjects


def verify_single_file(filepath):
    """Verify a single .set file loads correctly in MNE."""
    if not HAS_MNE:
        return None, "MNE not installed"
    try:
        raw = mne.io.read_raw_eeglab(filepath, preload=False, verbose=False)
        info = {
            'n_channels': len(raw.ch_names),
            'sfreq': raw.info['sfreq'],
            'duration_s': raw.n_times / raw.info['sfreq'],
            'ch_names_sample': raw.ch_names[:5]
        }
        return info, None
    except Exception as e:
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(description='Verify TO-CS-02 EEG data')
    parser.add_argument('--data-dir', required=True,
                        help='Path to Sedation-RestingState directory')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"ERROR: Directory not found: {data_dir}")
        sys.exit(1)

    print("=" * 60)
    print("TO-CS-02: Data Verification")
    print("=" * 60)

    # 1. Check for LFS pointers
    print("\n[1] Checking for Git LFS pointers...")
    lfs_count = 0
    real_count = 0
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.endswith(('.set', '.fdt', '.mat')):
            if check_lfs_pointer(fpath):
                lfs_count += 1
                if lfs_count <= 3:
                    print(f"  LFS POINTER: {fname}")
            else:
                real_count += 1

    if lfs_count > 0:
        print(f"\n  WARNING: {lfs_count} files are Git LFS pointers!")
        print(f"  Run 'git lfs pull' to download actual data.")
        print(f"  Real data files: {real_count}")
        if real_count == 0:
            print("\n  FATAL: No actual EEG data found. Cannot proceed.")
            sys.exit(1)
    else:
        print(f"  OK: All {real_count} data files contain actual data")

    # 2. Find subjects
    print("\n[2] Finding subjects...")
    subjects = find_set_files(data_dir)
    print(f"  Found {len(subjects)} subjects: {sorted(subjects.keys())}")

    for sub_id, files in sorted(subjects.items()):
        conditions = len(files)
        print(f"  Sub-{sub_id}: {conditions} conditions")
        if conditions != 4:
            print(f"    WARNING: Expected 4 conditions, found {conditions}")

    # 3. Check datainfo.mat
    print("\n[3] Checking datainfo.mat...")
    mat_path = os.path.join(data_dir, 'datainfo.mat')
    if os.path.exists(mat_path):
        if check_lfs_pointer(mat_path):
            print("  WARNING: datainfo.mat is an LFS pointer")
        elif HAS_SCIPY:
            try:
                mat = scipy.io.loadmat(mat_path)
                print(f"  OK: datainfo.mat loaded, keys: {[k for k in mat.keys() if not k.startswith('_')]}")
            except Exception as e:
                print(f"  ERROR loading datainfo.mat: {e}")
        else:
            print("  scipy not installed, cannot read .mat")
    else:
        print("  WARNING: datainfo.mat not found")

    # 4. Verify one file with MNE
    print("\n[4] MNE verification...")
    if not HAS_MNE:
        print("  SKIP: MNE not installed. Run: pip install mne")
    else:
        # Pick first available .set file
        test_file = None
        for sub_id, files in sorted(subjects.items()):
            for fname in files:
                fpath = os.path.join(data_dir, fname)
                if not check_lfs_pointer(fpath):
                    test_file = fpath
                    break
            if test_file:
                break

        if test_file:
            print(f"  Testing: {os.path.basename(test_file)}")
            info, error = verify_single_file(test_file)
            if info:
                print(f"  Channels: {info['n_channels']}")
                print(f"  Sampling rate: {info['sfreq']} Hz")
                print(f"  Duration: {info['duration_s']:.1f} s")
                print(f"  Sample channels: {info['ch_names_sample']}")

                if info['n_channels'] < 80:
                    print(f"  WARNING: Expected ~91 channels, got {info['n_channels']}")
                if info['sfreq'] != 250.0:
                    print(f"  WARNING: Expected 250 Hz, got {info['sfreq']}")
            else:
                print(f"  ERROR: {error}")
        else:
            print("  SKIP: No real .set files available to test")

    # 5. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Subjects found:     {len(subjects)}")
    print(f"  Real data files:    {real_count}")
    print(f"  LFS pointers:       {lfs_count}")
    print(f"  MNE available:      {HAS_MNE}")
    print(f"  scipy available:    {HAS_SCIPY}")

    if lfs_count == 0 and real_count > 0:
        print("\n  STATUS: READY TO RUN PIPELINE")
        print("  Next: python run_pipeline.py --data-dir \"" + data_dir + "\"")
    elif lfs_count > 0 and real_count > 0:
        print("\n  STATUS: PARTIAL DATA (some LFS pointers remain)")
    else:
        print("\n  STATUS: NO DATA (all LFS pointers)")
        print("  Fix: cd to repo and run 'git lfs pull'")


if __name__ == '__main__':
    main()
