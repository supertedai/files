#!/usr/bin/env python3
"""
TO-CS-02d: Delta-Free Within-N2 Product Test
=============================================
Same as TO-CS-02c but stop rule uses Ω(4-40) instead of Ω(full).

Pre-registered: 2026-02-08
Motivation: TO-CS-02c showed corr(Ω_4-40, κ) median = 0.556 (below 0.6 threshold)
            while corr(Ω_full, κ) = 0.704 (above). Delta power drives the correlation.

Changes from TO-CS-02c:
  1. Stop rule: corr(Ω_4-40, κ) instead of corr(Ω_full, κ)
  2. Primary product: C = Ω(4-40) × κ (not Ω_full × κ)
  3. All comparisons use delta-free Ω as primary

Everything else identical (same data, same TE, same epochs, same scoring).

Usage: python to_cs_02d_pipeline.py
"""

import os, sys, json, glob, warnings, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.signal import welch, butter, filtfilt, hilbert

warnings.filterwarnings('ignore')

DATA_DIR = "./anphy_sleep_data"
RESULTS_DIR = "./results_02d"

# LOCKED PARAMETERS (inherited)
LOCKED = {
    "epoch_length_sec": 10, "epoch_reject_uv": 300,
    "min_epochs_n2_total": 50, "min_epochs_pre_arousal": 5,
    "min_epochs_stable_n2": 20,
    "se_nperseg": 2048, "se_noverlap_frac": 0.5,
    "band_full": (0.5, 45.0), "band_delta_free": (4.0, 40.0),
    "te_k": 3, "te_bins": 6,
    "n_bootstrap": 10000, "bootstrap_seed": 42,
    "alpha": 0.05, "n_comparisons": 3, "alpha_corrected": 0.05 / 3,  # 3 primary comparisons
    "min_subjects": 5,
    # TO-CS-02d specific
    "stop_rule_corr_threshold": 0.6,
    "stop_rule_omega_band": "delta_free",  # KEY CHANGE from 02c
}

STAGE_MAP = {
    'W': 'Wake', 'R': 'REM', 'N1': 'N1', 'N2': 'N2', 'N3': 'N3',
    '0': 'Wake', '1': 'N1', '2': 'N2', '3': 'N3', '4': 'N3', '5': 'REM',
    'WAKE': 'Wake', 'REM': 'REM', 'Wake': 'Wake',
}

EXCLUDE_PATTERNS = [
    "EOG", "EMG", "ECG", "EKG", "ZY", "SO1", "SO2",
    "RLEG", "LLEG", "ChEMG", "Status", "STI",
    "FT11", "FT12", "TP11", "TP12", "F11", "F12", "P11", "P12",
]

ROI_DEFINITIONS = {
    "F_L":  ["Fp1", "AF3", "AF7", "F3", "F5", "F7", "F1", "F9"],
    "F_R":  ["Fp2", "AF4", "AF8", "F4", "F6", "F8", "F2", "FZ", "F10", "FPZ", "AFZ"],
    "C_L":  ["C3", "C5", "C1", "FC3", "FC5", "FC1", "CP3", "CP1", "T3", "T9", "FT9", "FT7", "TP9", "TP7"],
    "C_R":  ["C4", "C6", "C2", "FC4", "FC6", "FC2", "CP4", "CP2", "CZ", "FCZ", "CPZ", "T4", "T10", "FT8", "FT10", "TP8", "TP10"],
    "P_L":  ["P3", "P5", "P7", "P1", "CP5", "PO3", "PO7", "T5", "P9"],
    "P_R":  ["P4", "P6", "P8", "P2", "CP6", "PO4", "PO8", "PZ", "POZ", "T6", "P10"],
    "O_L":  ["O1", "PO7", "P7"],
    "O_R":  ["O2", "PO8", "OZ", "P8"],
}

def log(msg): print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# === HELPERS (identical to 02c) ===

def find_edf_files(d):
    f = []
    for p in ["**/*.edf", "**/*.EDF"]: f.extend(glob.glob(os.path.join(d, p), recursive=True))
    return sorted(set(f))

def is_eeg(ch):
    return not any(p.lower() in ch.lower() for p in EXCLUDE_PATTERNS)

def map_rois(ch_names):
    def norm(n): return n.strip().replace('.','').replace('-','').replace(' ','').upper()
    ch_n = {norm(c): i for i, c in enumerate(ch_names)}
    rm, mc = {}, {}
    for rn, rcs in ROI_DEFINITIONS.items():
        idx = [ch_n[norm(r)] for r in rcs if norm(r) in ch_n]
        if len(idx) >= 2: rm[rn] = idx; mc[rn] = [ch_names[i] for i in idx]
    return rm, mc

def load_scoring(p):
    sc = []
    with open(p) as f:
        for l in f:
            l = l.strip()
            if not l: continue
            for pt in (l.split('\t') if '\t' in l else l.split()):
                if pt.strip() in STAGE_MAP: sc.append(STAGE_MAP[pt.strip()]); break
    return sc

def load_edf(fp):
    import mne; mne.set_log_level('WARNING')
    try: raw = mne.io.read_raw_edf(fp, preload=True, verbose=False)
    except: return None, None
    ren = {c: c.replace('-Ref','') for c in raw.ch_names if c.endswith('-Ref')}
    if ren: raw.rename_channels(ren)
    raw.pick_channels([c for c in raw.ch_names if is_eeg(c)])
    if raw.info['sfreq'] != 250: raw.resample(250)
    raw.filter(0.5, 45.0, verbose=False)
    return raw, raw.ch_names

def spectral_entropy(sig, sfreq, band):
    nperseg = min(LOCKED["se_nperseg"], len(sig))
    f, p = welch(sig, fs=sfreq, nperseg=nperseg, noverlap=int(nperseg*LOCKED["se_noverlap_frac"]))
    m = (f >= band[0]) & (f <= band[1]); pb = p[m]
    if pb.sum() == 0: return 0.0
    pn = pb/pb.sum(); pn = pn[pn>0]
    H = -np.sum(pn*np.log2(pn)); Hm = np.log2(len(pn))
    return H/Hm if Hm > 0 else 0.0

def discretise(s, nb):
    mn, mx = s.min(), s.max()
    if mx == mn: return np.zeros(len(s), dtype=int)
    return np.digitize(s, np.linspace(mn, mx, nb+1)[1:-1]).astype(int)

def te_pair(src, tgt, k, nb):
    try:
        from pyinform import transfer_entropy as tf
        return tf(discretise(src,nb), discretise(tgt,nb), k=k)
    except ImportError: return _mte(src, tgt, k, nb)
    except: return 0.0

def _mte(src, tgt, k, nb):
    from collections import Counter
    sd, td = discretise(src,nb), discretise(tgt,nb); n=len(sd)
    if n <= k+1: return 0.0
    jxyz,jyz,jyo,jy = Counter(),Counter(),Counter(),Counter()
    for t in range(k,n-1):
        yf=td[t+1]; yp=tuple(td[t-k+1:t+1]); xp=tuple(sd[t-k+1:t+1])
        jxyz[(yf,yp,xp)]+=1; jyz[(yf,yp)]+=1; jyo[yp]+=1; jy[(yp,xp)]+=1
    tot=sum(jxyz.values())
    if tot==0: return 0.0
    te=0.0
    for (yf,yp,xp),c in jxyz.items():
        px=c/tot; pa=c/jy[(yp,xp)] if jy[(yp,xp)]>0 else 0; pb=jyz[(yf,yp)]/jyo[yp] if jyo[yp]>0 else 0
        if pa>0 and pb>0: te+=px*np.log2(pa/pb)
    return max(te,0.0)

def classify_n2(scoring, meta):
    buf = 2; nxt = ["N1", "Wake"]; ns = len(scoring)
    pre, stb, oth = [], [], []
    for ei, (si, sub) in enumerate(meta):
        is_pre = sub == 2 and si+1 < ns and scoring[si+1] in nxt
        is_stb = all(0 <= si+o < ns and scoring[si+o] == "N2" for o in range(-buf, buf+1))
        if is_pre: pre.append(ei)
        elif is_stb: stb.append(ei)
        else: oth.append(ei)
    return {"pre_arousal": pre, "stable": stb, "other": oth}

# === PER-EPOCH COMPUTATION ===

def compute_epochs(roi_sigs, sfreq):
    rn = list(roi_sigs.keys()); nr = len(rn)
    ne = roi_sigs[rn[0]].shape[0]
    o_f = np.zeros(ne); o_d = np.zeros(ne); kp = np.zeros(ne)
    tp2c = np.zeros(ne); tc2p = np.zeros(ne)

    for ep in range(ne):
        if ep % 100 == 0 and ep > 0: log(f"      epoch {ep}/{ne}...")
        sf, sd = [], []
        for r in rn:
            sf.append(spectral_entropy(roi_sigs[r][ep], sfreq, LOCKED["band_full"]))
            sd.append(spectral_entropy(roi_sigs[r][ep], sfreq, LOCKED["band_delta_free"]))
        o_f[ep] = np.mean(sf); o_d[ep] = np.mean(sd)
        tv, p2c, c2p = [], [], []
        for i, s in enumerate(rn):
            for j, t in enumerate(rn):
                if i==j: continue
                v = te_pair(roi_sigs[s][ep], roi_sigs[t][ep], LOCKED["te_k"], LOCKED["te_bins"])
                tv.append(v)
                if s.startswith("P") and t.startswith("C"): p2c.append(v)
                elif s.startswith("C") and t.startswith("P"): c2p.append(v)
        kp[ep] = np.mean(tv)
        tp2c[ep] = np.mean(p2c) if p2c else 0
        tc2p[ep] = np.mean(c2p) if c2p else 0

    return {
        "omega_full": o_f, "omega_df": o_d, "kappa": kp,
        "C_full": o_f*kp, "C_df": o_d*kp, "C_add": o_d+kp,
        "te_p2c": tp2c, "te_c2p": tc2p, "a_pc": tp2c-tc2p,
    }

# === STATISTICS ===

def cohens_d(a, b):
    n1,n2=len(a),len(b)
    if n1<2 or n2<2: return 0.0
    sp=np.sqrt(((n1-1)*np.var(a,ddof=1)+(n2-1)*np.var(b,ddof=1))/(n1+n2-2))
    return (np.mean(a)-np.mean(b))/sp if sp>0 else 0.0

def boot_dd(a1,a2,b1,b2,nb,seed):
    rng=np.random.RandomState(seed)
    dd=[]
    for _ in range(nb):
        i1=rng.choice(len(a1),len(a1),True); i2=rng.choice(len(a2),len(a2),True)
        j1=rng.choice(len(b1),len(b1),True); j2=rng.choice(len(b2),len(b2),True)
        dd.append(cohens_d(a1[i1],a2[i2])-cohens_d(b1[j1],b2[j2]))
    dd=np.array(dd)
    return {"delta_d":np.mean(dd),"ci_low":np.percentile(dd,2.5),"ci_high":np.percentile(dd,97.5),
            "p_value":np.mean(dd<=0),"passes":np.percentile(dd,2.5)>0 and np.mean(dd<=0)<LOCKED["alpha_corrected"]}

# === MAIN ===

def run_subject(edf, sid):
    import mne
    sd = os.path.dirname(edf)
    txts = glob.glob(os.path.join(sd,"*.txt"))
    if not txts: log("  SKIP: No scoring"); return None
    scoring = load_scoring(txts[0])
    if not scoring: return None
    n2c = sum(1 for s in scoring if s=="N2")
    log(f"  N2 scoring epochs: {n2c}")
    raw, chn = load_edf(edf)
    if raw is None: return None
    sfreq = raw.info['sfreq']
    rm, mc = map_rois(chn)
    if len(rm)<4: log("  SKIP: <4 ROIs"); return None

    data = raw.get_data(); se_s = int(30*sfreq); sub_s = int(10*sfreq)
    eps, meta = [], []
    for si, st in enumerate(scoring):
        if st != "N2": continue
        for sub in range(3):
            s = si*se_s + sub*sub_s; e = s+sub_s
            if e > data.shape[1]: continue
            ep = data[:, s:e]
            if np.max(np.ptp(ep,axis=1)*1e6) <= LOCKED["epoch_reject_uv"]:
                eps.append(ep); meta.append((si,sub))
    if not eps: return None
    eps = np.array(eps)
    log(f"  Clean N2 epochs: {len(eps)}")
    if len(eps) < LOCKED["min_epochs_n2_total"]: return None

    cl = classify_n2(scoring, meta)
    np_a, ns_a = len(cl["pre_arousal"]), len(cl["stable"])
    log(f"  Pre-arousal: {np_a}, Stable: {ns_a}")
    if np_a < LOCKED["min_epochs_pre_arousal"] or ns_a < LOCKED["min_epochs_stable_n2"]: return None

    roi_sigs = {rn: np.mean(eps[:,ci,:],axis=1) for rn,ci in rm.items()}
    log(f"  Computing per-epoch measures ({len(eps)} epochs)...")
    measures = compute_epochs(roi_sigs, sfreq)

    # Within-N2 correlations
    r_full, _ = stats.pearsonr(measures["omega_full"], measures["kappa"])
    r_df, _ = stats.pearsonr(measures["omega_df"], measures["kappa"])
    log(f"  corr(Ω_full, κ) = {r_full:.3f}")
    log(f"  corr(Ω_4-40, κ) = {r_df:.3f}  ← STOP RULE VARIABLE")

    # Report groups
    for gn, gi in [("pre_arousal", cl["pre_arousal"]), ("stable", cl["stable"])]:
        if len(gi) < 3: continue
        ix = np.array(gi)
        log(f"    {gn}(N={len(ix)}): Ω_df={np.mean(measures['omega_df'][ix]):.4f}, "
            f"κ={np.mean(measures['kappa'][ix]):.4f}, C_df={np.mean(measures['C_df'][ix]):.4f}, "
            f"TE(P→C)={np.mean(measures['te_p2c'][ix]):.4f}, A_PC={np.mean(measures['a_pc'][ix]):.4f}")

    return {"subject_id": sid, "n_epochs": len(eps), "n_pre": np_a, "n_stable": ns_a,
            "corr_full": r_full, "corr_df": r_df, "classification": cl,
            "measures": {k: v.tolist() for k,v in measures.items()}}


def score_test(results):
    log(f"\n{'='*60}")
    log("SCORING — TO-CS-02d DECISION TREE")
    log(f"{'='*60}")
    res = [r for r in results if r is not None]
    n = len(res)
    log(f"Usable subjects: {n}")
    if n < LOCKED["min_subjects"]:
        log(f"STOP: N={n} < {LOCKED['min_subjects']} → I"); return "I", {}

    # Step 1: Stop rule on Ω(4-40) — THE KEY DIFFERENCE
    log(f"\n--- Step 1: Within-N2 corr(Ω_4-40, κ) ---")
    corrs = [r["corr_df"] for r in res]
    med = np.median(corrs)
    for r in res: log(f"  {r['subject_id']}: corr(Ω_4-40,κ) = {r['corr_df']:.3f}")
    log(f"  Median: {med:.3f} (threshold: {LOCKED['stop_rule_corr_threshold']})")
    if med > LOCKED["stop_rule_corr_threshold"]:
        log(f"  STOP → I"); return "I", {"median_corr": med, "corrs": corrs}
    log(f"  PASS — Dissociation confirmed with delta-free Omega")

    # Collect pooled data
    pre_d, stb_d = {k:[] for k in ["omega_df","kappa","C_df","C_add","te_p2c","a_pc"]}, \
                   {k:[] for k in ["omega_df","kappa","C_df","C_add","te_p2c","a_pc"]}
    for r in res:
        m = {k: np.array(v) for k,v in r["measures"].items()}
        pi, si = np.array(r["classification"]["pre_arousal"]), np.array(r["classification"]["stable"])
        for k in pre_d: pre_d[k].extend(m[k][pi].tolist()); stb_d[k].extend(m[k][si].tolist())
    for k in pre_d: pre_d[k]=np.array(pre_d[k]); stb_d[k]=np.array(stb_d[k])

    np_t, ns_t = len(pre_d["omega_df"]), len(stb_d["omega_df"])
    log(f"\n--- Step 2: Effect sizes (pre-arousal vs stable) ---")
    log(f"  Pooled: {np_t} pre-arousal, {ns_t} stable epochs")
    es = {}
    for k in ["omega_df","kappa","C_df","C_add","te_p2c","a_pc"]:
        d = cohens_d(pre_d[k], stb_d[k]); es[k] = d
        log(f"  d({k}) = {d:.3f} (pre={np.mean(pre_d[k]):.5f}, stb={np.mean(stb_d[k]):.5f})")

    # Comparison (a): C_df vs omega_df
    log(f"\n--- Comparison (a): C(4-40) vs Omega(4-40) ---")
    ca = boot_dd(pre_d["C_df"],stb_d["C_df"],pre_d["omega_df"],stb_d["omega_df"],
                 LOCKED["n_bootstrap"],LOCKED["bootstrap_seed"])
    log(f"  delta_d={ca['delta_d']:.3f} CI=[{ca['ci_low']:.3f},{ca['ci_high']:.3f}] p={ca['p_value']:.4f} pass={ca['passes']}")

    # Comparison (b): C_df vs kappa
    log(f"\n--- Comparison (b): C(4-40) vs kappa ---")
    cb = boot_dd(pre_d["C_df"],stb_d["C_df"],pre_d["kappa"],stb_d["kappa"],
                 LOCKED["n_bootstrap"],LOCKED["bootstrap_seed"])
    log(f"  delta_d={cb['delta_d']:.3f} CI=[{cb['ci_low']:.3f},{cb['ci_high']:.3f}] p={cb['p_value']:.4f} pass={cb['passes']}")

    # Comparison (c): C_mult vs C_add
    log(f"\n--- Comparison (c): C_mult vs C_add ---")
    cc = boot_dd(pre_d["C_df"],stb_d["C_df"],pre_d["C_add"],stb_d["C_add"],
                 LOCKED["n_bootstrap"],LOCKED["bootstrap_seed"])
    log(f"  delta_d={cc['delta_d']:.3f} CI=[{cc['ci_low']:.3f},{cc['ci_high']:.3f}] p={cc['p_value']:.4f} pass={cc['passes']}")

    # Directional analysis (H2)
    log(f"\n--- Directional asymmetry (H2) ---")
    a_pc_pre = pre_d["a_pc"]; a_pc_stb = stb_d["a_pc"]
    d_apc = cohens_d(a_pc_pre, a_pc_stb)
    log(f"  d(A_PC) = {d_apc:.3f}")
    log(f"  pre-arousal A_PC = {np.mean(a_pc_pre):.5f} +/- {np.std(a_pc_pre):.5f}")
    log(f"  stable A_PC = {np.mean(a_pc_stb):.5f} +/- {np.std(a_pc_stb):.5f}")

    # TE(P->C) specifically
    d_p2c = cohens_d(pre_d["te_p2c"], stb_d["te_p2c"])
    log(f"  d(TE_P->C) = {d_p2c:.3f}")

    # Decision
    log(f"\n--- DECISION ---")
    a_pass = ca["passes"]; b_pass = cb["passes"]
    if a_pass and b_pass:
        if ca["delta_d"] >= 0.5: score = "C3"; log(f"  C beats both, delta_d>=0.5 -> C3")
        else: score = "C2"; log(f"  C beats both -> C2")
    elif a_pass or b_pass:
        score = "C1"; log(f"  C beats one component -> C1")
    else:
        score = "F1"; log(f"  C doesn't beat components -> F1")

    det = {"median_corr_df": med, "corrs": corrs, "effects": es,
           "comp_a": ca, "comp_b": cb, "comp_c": cc, "d_A_PC": d_apc, "d_TE_P2C": d_p2c}
    return score, det


def main():
    log("="*60)
    log("TO-CS-02d: Delta-Free Within-N2 Product Test")
    log("Pre-registered: 2026-02-08 | Stop rule: corr(Omega_4-40, kappa)")
    log("="*60)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    edfs = find_edf_files(DATA_DIR)
    log(f"Found {len(edfs)} EDF files")
    if not edfs: log("ERROR: No EDFs"); sys.exit(1)
    results = []
    for i, ef in enumerate(edfs):
        sid = Path(ef).stem
        log(f"\n{'='*40}\nSUBJECT {i+1}/{len(edfs)}: {sid}\n{'='*40}")
        results.append(run_subject(ef, sid))
    usable = [r for r in results if r]
    log(f"\n{'='*60}\nPIPELINE COMPLETE: {len(usable)}/{len(results)} usable")
    score, det = score_test(usable)
    summary = {"test_id":"TO-CS-02d","date":datetime.now().isoformat(),
               "n_subjects":len(usable),"score":score,"details":det}
    sf = os.path.join(RESULTS_DIR,"TO-CS-02d_summary.json")
    with open(sf,'w') as f: json.dump(summary,f,indent=2,default=lambda x:float(x) if isinstance(x,(np.floating,np.integer)) else x)
    log(f"\n{'='*60}\nFINAL SCORE: {score}\nSummary: {sf}\n{'='*60}")
    return score

if __name__=="__main__": main()
