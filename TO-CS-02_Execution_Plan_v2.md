# TO-CS-02: Execution Plan v2 — Updated with Preview Findings
**Date:** 2026-02-07  
**Version:** 2.0 (incorporates Ω preview results from freq_resting.mat)

---

## 1. WHAT WE ALREADY KNOW (before running κ)

### 1.1 Pipeline validated (sub-22)
Full chain works: BrainVision → MNE → 8 ROIs → Ω(PE/SE) + κ(TE) → C → dissociation check.  
corr(Ω, κ) = 0.16 at epoch level → stop rule PASSES.

### 1.2 Double dissociation in Ω (all 20 subjects)
At moderate sedation (same drug concentration):
- **Responsive (N=13):** Ω increases +0.054 (p=0.001)
- **Drowsy (N=7):** Ω decreases −0.063 (p=0.016)
- **Between-group:** Cohen's d = 1.54 (global), p = 0.011

### 1.3 Topography
Centro-parietal dominates. 18 of top 20 channels are P_L, C_R, P_R.
- Peak: E46 (P_L) d=1.94, dΔ=2.57
- Weakest: O1 d=0.47 (occipital)

### 1.4 Band robustness (CRITICAL)
| Band | Global d | E46 d | Interpretation |
|------|----------|-------|----------------|
| 0.5–45 Hz (full) | 1.54 | 1.94 | Strong — includes delta |
| 1–40 Hz | 1.38 | 1.90 | Strong |
| 4–40 Hz (no delta) | **0.56** | **1.14** | DROPS — delta drives much of the signal |
| 13–45 Hz (beta+gamma) | — | 1.72 | Moderate-strong |

**Implication:** Much of Ω_SE's discriminative power comes from delta-band redistribution, which is a *known* propofol LOC biomarker, not a novel EFC-C finding. The novel test is whether κ adds information *beyond* spectral power shift.

### 1.5 Subject classification (locked)

| Drowsy (N=7) | Correct at moderate |
|---------------|-------------------|
| Sub-02 | 3/40 |
| Sub-04 | 1/40 |
| Sub-05 | 6/40 |
| Sub-06 | 21/40 |
| Sub-08 | 0/40 |
| Sub-10 | 12/40 |
| Sub-17 | 15/40 |

---

## 2. LOCKED PREDICTIONS FOR κ (pre-registered before seeing TE data)

### Prediction κ-1: Parietal hub drives κ separation
- **κ_in(P_L)** = mean TE from all ROIs → P_L
- **κ_out(P_L)** = mean TE from P_L → all ROIs
- Hypothesis: κ_in(P_L) or κ_out(P_L) shows larger d (responsive vs drowsy) than κ averaged over all ROI pairs.

### Prediction κ-2: Top-down asymmetry
- **A_FP** = TE(F→P) − TE(P→F), similarly A_CP = TE(C→P) − TE(P→C)
- Hypothesis: |ΔA_FP| between groups is larger than symmetric κ change.

### Prediction κ-3: Occipital null
- κ involving O_L/O_R gives lower Δd than κ involving P_L/C_R.

### Prediction C-1: Product beats delta-free Ω
- C = Ω(4–40 Hz) × κ gives d > 0.56 (global delta-free Ω bar)
- This is the *primary novel test* — it asks whether integration adds beyond spectral power.

### Prediction C-2: Product vs full-band Ω
- C = Ω(full) × κ gives d > 1.54 (global full-band Ω bar)
- This is the *hard test* — full-band Ω already captures delta shift.

---

## 3. UPDATED EXECUTION SEQUENCE

### Phase 0: Download (BLOCKED — needs manual download)
Download `sedation-restingstate.zip` (3.44 GB) from:
https://www.repository.cam.ac.uk/handle/1810/252736

### Phase 1: Load & verify (30 min)
- Load all 20 subjects × 4 conditions
- Verify channel counts, epoch counts match expectations
- Extract hit rates from datainfo.mat
- Confirm drowsy/responsive classification matches our labels

### Phase 2: Compute Ω (15 min)
- Ω_PE (permutation entropy) on ROI-averaged signals
- Ω_SE (spectral entropy) at two bands: full (0.5–45) and delta-free (4–40)
- Statistical unit = subject × condition (mean over epochs)
- Validate against freq_resting.mat preview (should match)

### Phase 3: Compute κ — HIERARCHICAL (save time)

**Phase 3a: Targeted κ (30 min)**
- κ on 8 ROIs, BUT extract separately:
  - κ_global (mean all 56 pairs)
  - κ_in(P_L), κ_out(P_L)
  - κ_in(C_R), κ_out(C_R)
  - A_FP, A_CP (directional asymmetries)

**Phase 3b: Full 8-ROI κ (if 3a shows signal) (30 min)**
- Complete 56-pair TE matrix per subject × condition
- Quadrant analysis: Ω vs κ scatter

**Phase 3c: 32-channel robustness (if 3b shows signal) (~5 hours)**
- Include E46/E47/E87/E55 + frontal/central partners
- Only run if ROI-level κ shows d > 0.5

### Phase 4: Dissociation check (10 min)
- corr(Ω, κ) on subject × condition level (N=80 points)
- Stop rule: corr > 0.8 → Indeterminate

### Phase 5: Four-way comparison (30 min)
For each contrast (responsive vs drowsy at moderate):

| Measure | What it tests |
|---------|---------------|
| Ω_SE (full) | Baseline — already d=1.54 |
| Ω_SE (4–40 Hz) | Conservative — d=0.56 |
| κ_global | Integration alone |
| κ_P_L (in+out) | Parietal-specific integration |
| C = Ω(full) × κ | Product vs full Ω |
| C = Ω(4–40) × κ | Product vs delta-free Ω ← **KEY TEST** |
| C_add = Ω + κ | Additive control |

Bootstrap 95% CI on Δd for each comparison.

### Phase 6: Scoring (15 min)
Apply TO-CS-02 §6 criteria. Key question:
- Does C = Ω(4–40) × κ beat d = 0.56? (novel contribution)
- Does C = Ω(full) × κ beat d = 1.54? (hard test)

---

## 4. DECISION TREE

```
Phase 4: corr(Ω,κ) > 0.8?
  YES → Indeterminate, STOP
  NO ↓
Phase 5: Does C beat Ω(4-40)?
  NO → F1/F2 (κ adds nothing)
  YES ↓
Does C beat Ω(full)?
  NO → C1 (κ adds beyond spectral shift, but not beyond full Ω)
  YES ↓
Does C beat Ω(full) by Δd ≥ 0.5?
  NO → C2 (moderate support)
  YES → C3 (strong support for product hypothesis)
```

---

## 5. WHAT MAKES THE 4–40 Hz TEST THE REAL PRIZE

Full-band Ω_SE is largely driven by delta power redistribution — a well-known propofol LOC marker. If C = Ω × κ only beats full-band Ω, we haven't shown anything new.

But if C = Ω(4–40 Hz) × κ strongly separates conscious from unconscious while Ω(4–40) alone barely does (d=0.56), that means:

**Causal integration (κ) captures consciousness-related information that is NOT in the power spectrum.**

That would be a genuine contribution of TO-CS-02 and evidence that EFC-C's "both differentiation AND integration" claim has empirical substance beyond known spectral biomarkers.

---

## 6. COMPUTATIONAL BUDGET (unchanged from v1)

| Step | Time |
|------|------|
| Ω (all subjects) | < 5 min |
| κ at 8 ROIs (all subjects) | ~16 min |
| κ at 32 channels (robustness) | ~5 hours |
| Analysis + scoring | ~30 min |
| **Total (primary)** | **~1 hour** |
| **Total (with robustness)** | **~6 hours** |

---

*Execution plan v2 by: Claude (Symbiose) for Morten (EFC)*  
*Incorporates preview findings from freq_resting.mat analysis*  
*κ predictions locked before seeing TE data*
