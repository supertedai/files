# TO-CS-02: Preview Results — Ω_SE on All 20 Subjects
**Date:** 2026-02-07  
**Status:** PREVIEW (Ω only — κ awaits raw EEG)

---

## 1. WHAT WAS DONE

Extracted spectral entropy (Ω_SE) from pre-computed power spectra in `freq_resting.mat` for all 20 Chennu et al. 2016 subjects across 4 sedation conditions. Classified subjects as **responsive** (N=13, ≥24/40 correct at moderate) or **drowsy** (N=7, <24/40).

This uses **only Ω** — no κ (TE) computation possible without raw time-domain data.

---

## 2. KEY FINDING: DOUBLE DISSOCIATION

At **moderate sedation** (same drug concentration ~1000 µg/L):

| Group | Baseline Ω_SE | Moderate Ω_SE | Direction | Wilcoxon p |
|-------|---------------|---------------|-----------|------------|
| Responsive (N=13) | 0.762 | 0.816 | **↑ +0.054** | 0.0012 |
| Drowsy (N=7) | 0.782 | 0.720 | **↓ −0.063** | 0.0156 |

**Between-group at moderate: Cohen's d = 1.54, p = 0.011**

Same drug → opposite Ω direction → consciousness determines the response, not pharmacology.

---

## 3. REGIONAL BREAKDOWN

| Region | d (resp vs drowsy at moderate) | p | Responsive Δ | Drowsy Δ |
|--------|-------------------------------|---|--------------|----------|
| Central | **1.72** | 0.005 | +0.042 | −0.073 |
| Parietal | 1.56 | 0.011 | +0.053 | −0.069 |
| Frontal | 1.52 | 0.009 | +0.061 | −0.053 |
| Occipital | 1.23 | 0.097 | +0.054 | −0.057 |

Central cortex shows the strongest dissociation. Occipital is weakest (marginal significance).

---

## 4. SUBJECT CLASSIFICATION

| Subject | Moderate correct (/40) | Group |
|---------|----------------------|-------|
| Sub-01 | 39 | Responsive |
| Sub-02 | 3 | Drowsy |
| Sub-03 | 33 | Responsive |
| Sub-04 | 1 | Drowsy |
| Sub-05 | 6 | Drowsy |
| Sub-06 | 21 | Drowsy |
| Sub-07 | 34 | Responsive |
| Sub-08 | 0 | Drowsy |
| Sub-09 | 35 | Responsive |
| Sub-10 | 12 | Drowsy |
| Sub-11 | 39 | Responsive |
| Sub-12 | 39 | Responsive |
| Sub-13 | 35 | Responsive |
| Sub-14 | 35 | Responsive |
| Sub-15 | 37 | Responsive |
| Sub-16 | 40 | Responsive |
| Sub-17 | 15 | Drowsy |
| Sub-18 | 37 | Responsive |
| Sub-19 | 38 | Responsive |
| Sub-20 | 38 | Responsive |

---

## 5. QUANTITATIVE BAR FOR PRODUCT HYPOTHESIS

Ω_SE alone achieves d = 1.54 (global) and d = 1.72 (central region).

**TO-CS-02 product hypothesis (C = Ω × κ) is supported only if:**
- C gives d > 1.54 at global level (exceeds Ω alone), OR
- C gives d > 1.72 at regional level (exceeds best single component)

**Specific prediction for κ:**
If fronto-parietal κ (transfer entropy between frontal and parietal ROIs) adds independent information about consciousness, then C_fronto-parietal = Ω × κ_FP should exceed d = 1.72.

---

## 6. WHAT THIS MEANS FOR EFC-C

The finding that Ω tracks **consciousness**, not drug dose, is directly predicted by EFC-C's framework where consciousness = high differentiation AND high causal closure. The differentiation component is now empirically grounded.

The open question is whether adding the causal closure component (κ) makes the measure better — or whether differentiation alone is sufficient.

---

## 7. BLOCKING ISSUE

κ requires raw time-domain EEG data. The `freq_resting.mat` file contains only power spectra (frequency domain). To complete TO-CS-02:

**Download `sedation-restingstate.zip` (3.44 GB) from:**  
https://www.repository.cam.ac.uk/handle/1810/252736

Pipeline is validated and ready to run immediately.

---

*Preview analysis by: Claude (Symbiose) for Morten (EFC)*  
*Data: Chennu et al. 2016 freq_resting.mat (all 20 subjects)*
