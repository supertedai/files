# TO-CS-02b: Local Execution Guide

## Quick Start

```bash
# 1. Install dependencies
pip install mne numpy scipy pyinform pandas

# 2. Create data folder and put your ANPHY-Sleep ZIPs there
mkdir -p anphy_sleep_data
# Copy/move your ZIP files: EPCTL21.zip, EPCTL23.zip, etc. -> anphy_sleep_data/

# 3. Edit DATA_DIR in the script (line ~40) if your path is different
#    Default: ./anphy_sleep_data

# 4. Run
python to_cs_02b_pipeline.py

# 5. Results appear in ./results/
#    - TO-CS-02b_summary.json  <- main output (score + all numbers)
#    - <subject>_results.json  <- per-subject detail
```

## What the Script Does

1. **Extracts ZIPs** in DATA_DIR
2. **Finds all EDF files** recursively
3. **Per subject:**
   - Loads EDF, picks EEG channels, resamples to 250 Hz, filters 0.5-45 Hz
   - Reads sleep stage annotations from EDF+
   - Cuts into 10s non-overlapping epochs, rejects >100 uV peak-to-peak
   - Balances epoch counts across stages
   - Maps channels to 8 ROIs (10-10 montage)
   - Computes Omega (spectral entropy, Welch) at full-band and delta-free
   - Computes kappa (transfer entropy, k=3, 6 bins) for all ROI pairs
   - Computes C = Omega x kappa, directed edges (P->C, C->P), asymmetries
4. **Group-level scoring:**
   - Dissociation check: corr(Omega, kappa)
   - Bootstrap delta_d for all pre-registered comparisons
   - Applies decision tree -> final score (C3/C2/C1/I/F1/F2)

## Troubleshooting

### "No sleep stage annotations found"
Your EDF files may use non-standard annotation labels. The script prints available annotations. Add any new labels to `STAGE_MAP` at line ~80.

### "Too few ROIs"
Your montage may use different channel names. The script auto-matches case-insensitively, but if channels use unusual naming (e.g., "EEG Fp1-REF"), you may need to strip prefixes. Check the channel list printed at load time.

### pyinform not available
The script includes a manual TE fallback. Results will be slightly different but functionally equivalent. For exact replication, install pyinform:
```bash
pip install pyinform
```

### Runtime
- ~5-15 minutes per subject (depends on epoch count and CPU)
- 5 subjects: ~30-60 minutes
- 10 subjects: ~1-2 hours

## Parameters (LOCKED -- do not modify)

All parameters in the `LOCKED` dict are pre-registered. The only thing you should edit is `DATA_DIR` at the top of the script.

## After Running

Upload `results/TO-CS-02b_summary.json` to the chat for interpretation and scoring.
