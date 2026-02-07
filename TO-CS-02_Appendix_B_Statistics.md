# TO-CS-02: Appendix B — Locked Statistical Procedure & Regional C Definitions
**Date locked:** 2026-02-07  
**Status:** PRE-REGISTERED (before any κ/TE data computed on full dataset)

---

## B.1 BOOTSTRAP Δd PROCEDURE

To determine whether measure A "beats" measure B in separating responsive from drowsy subjects:

| Parameter | Value |
|-----------|-------|
| N_bootstrap | 10,000 |
| Random seed | 42 (fixed) |
| Resampling | Within-group, paired (same subject index across measures) |
| CI method | Percentile, 95% |
| p-value | One-tailed: proportion of bootstrap Δd ≤ 0 |

**"Beats" criterion:**
- Δd = d(A) − d(B) > 0, AND
- Bootstrap 95% CI lower bound > 0

**For C3 scoring:** additionally Δd ≥ 0.5 or ΔAUC ≥ 0.05

---

## B.2 PRIMARY COMPARISONS (ordered)

| ID | Comparison | What it tests | Significance threshold |
|----|-----------|---------------|----------------------|
| **a** | C = Ω(4–40) × κ vs Ω(4–40) | κ adds beyond delta-free spectral power | p < 0.0125 |
| **b** | C = Ω(4–40) × κ vs κ alone | Ω adds beyond integration alone | p < 0.0125 |
| **c** | C = Ω(full) × κ vs Ω(full) | Product beats full spectral entropy | p < 0.0125 |
| **d** | C_add = Ω + κ vs C = Ω × κ | Multiplicative vs additive combination | p < 0.0125 |

**Multiple comparison correction:** Bonferroni, α = 0.05/4 = 0.0125 per comparison.

**Primary endpoint:** Comparison (a). This is the KEY NOVEL TEST.  
If (a) passes but (c) fails → κ captures information beyond spectral power shift, but not beyond full Ω. Score: C1.  
If (a) and (c) both pass → strong support. Score: C2 or C3 depending on Δd.

---

## B.3 REGIONAL C DEFINITIONS

### Level 1: ROI-level (primary analysis)

| Measure | Definition | Expected d |
|---------|-----------|------------|
| C_global | mean(Ω across 8 ROIs) × mean(κ across 56 pairs) | To be determined |
| C_PL | Ω(P_L) × κ_hub(P_L) | Predicted highest |
| C_CR | Ω(C_R) × κ_hub(C_R) | Predicted second |

Where:
- **Ω(P_L)** = mean spectral entropy (4–40 Hz) across channels in P_L ROI
- **κ_hub(P_L)** = mean of all TE pairs involving P_L (both directions):  
  κ_hub(P_L) = mean(κ_in(P_L), κ_out(P_L))  
  κ_in(P_L) = mean TE from {F_L, F_R, C_L, C_R, P_R, O_L, O_R} → P_L  
  κ_out(P_L) = mean TE from P_L → {F_L, F_R, C_L, C_R, P_R, O_L, O_R}

### Level 2: Directed (secondary analysis)

| Measure | Definition |
|---------|-----------|
| C_topdown | Ω(P_L) × mean[TE(F_L→P_L), TE(F_R→P_L)] |
| C_bottomup | Ω(P_L) × mean[TE(P_L→F_L), TE(P_L→F_R)] |
| A_FP | C_topdown − C_bottomup |

**Hypothesis:** |A_FP| is larger between groups than symmetric κ_hub(P_L).

### Level 3: Negative control

| Measure | Definition | Expected |
|---------|-----------|----------|
| C_occipital | Ω(O_L) × κ_hub(O_L) | Weakest separation |

If C_occipital shows d comparable to C_PL → topographic specificity claim fails.

---

## B.4 QUANTITATIVE BARS (from Ω preview)

| Bar | Source | d value | Role |
|-----|--------|---------|------|
| Global Ω(full) | All 20 subjects, 0.5–45 Hz | 1.54 | Hard bar for C(full) |
| Global Ω(4–40) | All 20 subjects, 4–40 Hz | 0.56 | **Primary bar for novel test** |
| Peak channel Ω(full) | E46 (P_L) | 1.94 | Upper bound |
| Peak channel Ω(4–40) | E46 (P_L) | 1.14 | Regional bar |
| Peak dΔ | E46, moderate−baseline | 2.57 | Change sensitivity bar |

---

## B.5 DECISION TREE

```
1. Compute corr(Ω, κ) at subject×condition level (N=80)
   │
   ├─ corr > 0.8 → STOP: Indeterminate (dataset unsuitable)
   │
   └─ corr ≤ 0.8 → Continue
       │
       2. Comparison (a): C = Ω(4-40)×κ vs Ω(4-40)
       │
       ├─ Δd ≤ 0 or CI includes 0 → F1 (κ adds nothing novel)
       │
       └─ Δd > 0, CI excludes 0, p < 0.0125
           │
           3. Comparison (c): C = Ω(full)×κ vs Ω(full)
           │
           ├─ Fails → C1 (κ adds beyond spectral shift only)
           │
           └─ Passes
               │
               4. Δd ≥ 0.5?
               │
               ├─ No → C2 (moderate support for product)
               │
               └─ Yes → C3 (strong support for product)
```

---

## B.6 WHAT CANNOT BE CHANGED AFTER SEEING κ DATA

The following are LOCKED and may not be modified post-hoc:

1. Subject classification (responsive/drowsy threshold: 24/40)
2. ROI definitions (coordinate-based quantile split, roi_mapping.json)
3. Ω computation method (spectral entropy, Welch PSD, normalised)
4. κ computation method (pyinform TE, k=3, 6 bins)
5. Frequency bands: full (0.5–45 Hz) and delta-free (4–40 Hz)
6. Bootstrap parameters (N=10000, seed=42, percentile CI)
7. Multiple comparison correction (Bonferroni, 4 tests, α=0.0125)
8. Regional C definitions (C_PL, C_topdown, C_bottomup, C_occipital)
9. All κ predictions (κ-1 through κ-3, C-1, C-2)
10. Decision tree and scoring criteria

---

*Locked by: Claude (Symbiose) for Morten (EFC)*  
*All predictions registered in Symbiose Knowledge Graph*  
*No κ/TE values have been computed on the full 20-subject dataset at time of locking*
