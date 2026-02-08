#!/usr/bin/env python3
"""
SUBJECT IDENTITY AUDIT
======================
Compare subject IDs in:
  1. freq_resting.mat (preview source) — if available
  2. .set files in Sedation-RestingState folder (pipeline source)
  3. datainfo.mat (hit rates / classification source)

Output: which subjects are in each, overlap, and classification.
"""

import os
import sys
import glob
import numpy as np
from pathlib import Path

DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "sedation-restingstate/Sedation-RestingState"

print("=" * 70)
print("SUBJECT IDENTITY AUDIT")
print("=" * 70)

# ============================================================
# 1. INSPECT freq_resting.mat
# ============================================================
print("\n--- freq_resting.mat ---")

mat_paths = glob.glob(os.path.join(DATA_DIR, "**/freq_resting*"), recursive=True)
if not mat_paths:
    mat_paths = glob.glob("**/freq_resting*", recursive=True)

if mat_paths:
    print(f"Found: {mat_paths[0]}")
    mat_path = mat_paths[0]

    try:
        import scipy.io as sio
        mat = sio.loadmat(mat_path)
        print(f"\nKeys: {[k for k in mat.keys() if not k.startswith('__')]}")
        for k, v in mat.items():
            if k.startswith('__'):
                continue
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                if v.dtype.names:
                    print(f"    fields: {v.dtype.names}")
                if 'sub' in k.lower() or 'id' in k.lower() or 'label' in k.lower():
                    print(f"    values: {v.flatten()[:30]}")
    except NotImplementedError:
        print("scipy can't read (v7.3+), trying h5py...")

    try:
        import h5py
        with h5py.File(mat_path, 'r') as f:
            print(f"\nHDF5 keys: {list(f.keys())}")
            for k in f.keys():
                ds = f[k]
                if hasattr(ds, 'shape'):
                    print(f"  {k}: shape={ds.shape}, dtype={ds.dtype}")
                    if ds.dtype == h5py.ref_dtype:
                        flat = list(ds[()].flat)
                        print(f"    {len(flat)} references")
                        for i, ref in enumerate(flat[:5]):
                            try:
                                target = f[ref]
                                if hasattr(target, 'shape') and target.shape:
                                    chars = target[:]
                                    if chars.dtype.kind in ('u', 'i') and chars.max() < 200:
                                        label = ''.join(chr(c) for c in chars.flat if 0 < c < 128)
                                        print(f"    [{i}]: '{label}'")
                            except:
                                pass
                    elif ds.ndim <= 2 and ds.size < 100:
                        try:
                            print(f"    values: {ds[()].flatten()[:20]}")
                        except:
                            pass
                if isinstance(ds, h5py.Group):
                    print(f"  {k}: group, subkeys={list(ds.keys())[:10]}")

            # Search for subject identifiers
            print("\n--- Searching for subject identifiers ---")
            def search(group, prefix=""):
                for k in group.keys():
                    item = group[k]
                    full_key = f"{prefix}/{k}" if prefix else k
                    if any(s in k.lower() for s in ['sub', 'id', 'label', 'name', 'part', 'info']):
                        print(f"  CANDIDATE: {full_key}")
                        if hasattr(item, 'shape'):
                            print(f"    shape={item.shape}, dtype={item.dtype}")
                    if isinstance(item, h5py.Group) and len(prefix.split('/')) < 3:
                        search(item, full_key)
            search(f)

            # Check multi-dim arrays
            for k in f.keys():
                ds = f[k]
                if hasattr(ds, 'shape') and len(ds.shape) >= 3:
                    print(f"\n  Multi-dim '{k}': shape={ds.shape}")
                    if len(ds.shape) == 4:
                        print(f"    dim0={ds.shape[0]} (subjects?), dim1={ds.shape[1]} (conditions?), "
                              f"dim2={ds.shape[2]} (channels?), dim3={ds.shape[3]} (frequencies?)")
    except Exception as e:
        print(f"h5py error: {e}")
else:
    print("freq_resting.mat NOT FOUND in dataset")
    print("(This is expected — it was from FieldTrip, not Cambridge)")

# ============================================================
# 2. LIST .set FILES AND EXTRACT SUBJECT IDs
# ============================================================
print("\n" + "=" * 70)
print("--- .set files ---")
print("=" * 70)

set_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.set")))
print(f"\nFound {len(set_files)} .set files")

# Group by subject
from collections import defaultdict
subjects = defaultdict(list)
for f in set_files:
    basename = os.path.basename(f)
    parts = basename.split('-')
    if len(parts) >= 2:
        sub_id = parts[0]
        subjects[sub_id].append(basename)

print(f"\nSubjects ({len(subjects)}):")
for sub_id in sorted(subjects.keys()):
    files = subjects[sub_id]
    print(f"  Sub-{sub_id}: {len(files)} files")
    for fn in files:
        print(f"    {fn}")

# ============================================================
# 3. INSPECT datainfo.mat
# ============================================================
print("\n" + "=" * 70)
print("--- datainfo.mat (classification source) ---")
print("=" * 70)

info_path = os.path.join(DATA_DIR, "datainfo.mat")
if not os.path.exists(info_path):
    info_paths = glob.glob(os.path.join(DATA_DIR, "**/datainfo*"), recursive=True)
    if info_paths:
        info_path = info_paths[0]
    else:
        info_path = None

if info_path:
    print(f"Found: {info_path}")

    # Check if LFS pointer
    try:
        with open(info_path, 'r', encoding='utf-8') as f:
            first = f.readline()
        if first.startswith('version https://git-lfs'):
            print("WARNING: This is a Git LFS pointer, not actual data!")
            print(f"Content: {first.strip()}")
        else:
            print("File is actual data (not LFS pointer)")
    except UnicodeDecodeError:
        print("File is binary (actual data)")

    # Try scipy first (older MATLAB format)
    try:
        import scipy.io as sio
        info = sio.loadmat(info_path)
        print(f"\nKeys (scipy): {[k for k in info.keys() if not k.startswith('__')]}")
        for k, v in info.items():
            if k.startswith('__'):
                continue
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                if v.dtype.names:
                    print(f"    fields: {v.dtype.names}")
                if v.size < 200:
                    flat = v.flatten()
                    print(f"    values: {flat[:30]}")
                    if len(flat) > 30:
                        print(f"    ... ({len(flat)} total)")
            else:
                print(f"  {k}: type={type(v)}")
    except NotImplementedError:
        print("scipy can't read (v7.3+)")
    except Exception as e:
        print(f"scipy error: {e}")

    # Try h5py (MATLAB v7.3)
    try:
        import h5py
        with h5py.File(info_path, 'r') as f:
            print(f"\nHDF5 keys: {list(f.keys())}")

            def dump_hdf5(group, prefix="", depth=0):
                if depth > 3:
                    return
                for k in group.keys():
                    item = group[k]
                    full = f"{prefix}/{k}" if prefix else k
                    if isinstance(item, h5py.Dataset):
                        print(f"  {'  '*depth}{full}: shape={item.shape}, dtype={item.dtype}")
                        if item.dtype == h5py.ref_dtype and item.size <= 50:
                            for i, ref in enumerate(item[()].flat):
                                if i >= 10:
                                    print(f"  {'  '*depth}  ... ({item.size} total)")
                                    break
                                try:
                                    target = f[ref]
                                    if hasattr(target, 'shape'):
                                        chars = target[:]
                                        if chars.dtype.kind in ('u', 'i') and chars.size < 100 and chars.max() < 200:
                                            label = ''.join(chr(c) for c in chars.flat if 0 < c < 128)
                                            print(f"  {'  '*depth}  [{i}]: '{label}'")
                                        else:
                                            print(f"  {'  '*depth}  [{i}]: shape={target.shape}, dtype={target.dtype}")
                                except:
                                    pass
                        elif item.size <= 50:
                            try:
                                vals = item[()].flatten()
                                print(f"  {'  '*depth}  values: {vals[:20]}")
                            except:
                                pass
                    elif isinstance(item, h5py.Group):
                        print(f"  {'  '*depth}{full}: GROUP")
                        dump_hdf5(item, full, depth+1)

            dump_hdf5(f)
    except Exception as e:
        print(f"h5py error: {e}")
else:
    print("datainfo.mat NOT FOUND")

# ============================================================
# 4. INSPECT .set FILE METADATA (HDF5)
# ============================================================
print("\n" + "=" * 70)
print("--- .set file internal metadata (first file per subject) ---")
print("=" * 70)

try:
    import h5py

    for sub_id in sorted(list(subjects.keys()))[:3]:  # First 3 subjects
        first_file = os.path.join(DATA_DIR, subjects[sub_id][0])

        # Check for LFS pointer
        try:
            with open(first_file, 'r', encoding='utf-8') as f:
                if f.readline().startswith('version https://git-lfs'):
                    print(f"\n  Sub-{sub_id}: LFS pointer (skip)")
                    continue
        except UnicodeDecodeError:
            pass

        try:
            with h5py.File(first_file, 'r') as f:
                eeg = f['EEG']
                nbchan = int(np.array(eeg['nbchan']).flat[0])
                pnts = int(np.array(eeg['pnts']).flat[0])
                sfreq = float(np.array(eeg['srate']).flat[0])
                trials = int(np.array(eeg['trials']).flat[0]) if 'trials' in eeg else 1
                xmin = float(np.array(eeg['xmin']).flat[0]) if 'xmin' in eeg else 0
                xmax = float(np.array(eeg['xmax']).flat[0]) if 'xmax' in eeg else 0

                print(f"\n  Sub-{sub_id} ({subjects[sub_id][0]}):")
                print(f"    nbchan={nbchan}, pnts={pnts}, sfreq={sfreq}")
                print(f"    trials={trials}, xmin={xmin}, xmax={xmax}")
                print(f"    duration = {pnts/sfreq:.1f}s ({pnts/sfreq/60:.1f}min)")

                # Check for events/epoch info
                if 'event' in eeg:
                    ev = eeg['event']
                    print(f"    events: keys={list(ev.keys()) if isinstance(ev, h5py.Group) else 'dataset'}")
                if 'epoch' in eeg:
                    ep = eeg['epoch']
                    print(f"    epoch: keys={list(ep.keys()) if isinstance(ep, h5py.Group) else 'dataset'}")

                # Check setname/comments for condition info
                for field in ['setname', 'comments', 'condition', 'session']:
                    if field in eeg:
                        ds = eeg[field]
                        if isinstance(ds, h5py.Dataset):
                            try:
                                if ds.dtype.kind in ('u', 'i') and ds.size < 200:
                                    val = ''.join(chr(c) for c in ds[()].flat if 0 < c < 128)
                                    print(f"    {field}: '{val}'")
                                elif ds.size < 50:
                                    print(f"    {field}: {ds[()].flatten()}")
                            except:
                                print(f"    {field}: shape={ds.shape}")
        except Exception as e:
            print(f"\n  Sub-{sub_id}: ERROR - {e}")

except ImportError:
    print("h5py not available")

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
