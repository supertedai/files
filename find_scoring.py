#!/usr/bin/env python3
"""
Diagnostic: Find sleep scoring files in ANPHY-Sleep data.
Sleep stages are NOT in the EDF annotations - they must be in separate files.
"""
import glob
import os

DATA_DIR = "./anphy_sleep_data"

print("=" * 70)
print("SEARCHING FOR ALL NON-EDF FILES IN DATA DIRECTORY")
print("=" * 70)

# Find ALL files recursively
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "**/*"), recursive=True))
all_files = [f for f in all_files if os.path.isfile(f)]

# Group by extension
from collections import defaultdict
by_ext = defaultdict(list)
for f in all_files:
    ext = os.path.splitext(f)[1].lower()
    by_ext[ext].append(f)

print(f"\nTotal files: {len(all_files)}")
print(f"\nFiles by extension:")
for ext, files in sorted(by_ext.items()):
    print(f"  {ext or '(no ext)'}: {len(files)} files")
    for f in files[:5]:
        size = os.path.getsize(f)
        print(f"    {os.path.relpath(f, DATA_DIR)}  ({size:,} bytes)")
    if len(files) > 5:
        print(f"    ... and {len(files)-5} more")

# Show contents of potential scoring files
scoring_exts = ['.tsv', '.csv', '.txt', '.xml', '.json', '.hyp', '.stg',
                '.scoring', '.sleep', '.ann', '.evt', '.rec', '.sta']

print(f"\n{'=' * 70}")
print("CONTENTS OF POTENTIAL SCORING FILES")
print(f"{'=' * 70}")

for ext in scoring_exts:
    if ext in by_ext:
        for f in by_ext[ext][:3]:  # Show first 3 of each type
            print(f"\n--- {os.path.relpath(f, DATA_DIR)} ---")
            try:
                with open(f, 'r', encoding='utf-8', errors='replace') as fh:
                    lines = fh.readlines()
                    for i, line in enumerate(lines[:30]):
                        print(f"  {i+1:4d} | {line.rstrip()}")
                    if len(lines) > 30:
                        print(f"  ... ({len(lines)} total lines)")
                        # Also show a few lines from the middle
                        mid = len(lines) // 2
                        print(f"  (lines {mid}-{mid+3} from middle:)")
                        for line in lines[mid:mid+3]:
                            print(f"       | {line.rstrip()}")
            except Exception as e:
                print(f"  ERROR reading: {e}")

# Also check for any files with "score", "stage", "hyp" in the name
print(f"\n{'=' * 70}")
print("FILES WITH 'score', 'stage', 'hyp', 'sleep' IN NAME")
print(f"{'=' * 70}")
keywords = ['score', 'stage', 'hyp', 'sleep', 'epoch', 'annotation']
for f in all_files:
    basename = os.path.basename(f).lower()
    if any(kw in basename for kw in keywords):
        size = os.path.getsize(f)
        print(f"  {os.path.relpath(f, DATA_DIR)}  ({size:,} bytes)")

# List directory structure (top 2 levels)
print(f"\n{'=' * 70}")
print("DIRECTORY STRUCTURE (top levels)")
print(f"{'=' * 70}")
for root, dirs, files in os.walk(DATA_DIR):
    depth = root.replace(DATA_DIR, '').count(os.sep)
    if depth <= 2:
        indent = '  ' * depth
        print(f"{indent}{os.path.basename(root)}/")
        if depth <= 1:
            for f in sorted(files)[:20]:
                print(f"{indent}  {f}")
            if len(files) > 20:
                print(f"{indent}  ... and {len(files)-20} more files")
