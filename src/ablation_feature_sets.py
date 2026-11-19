#!/usr/bin/env python3
"""Feature ablation for the learned substitution-risk score -- CPU ONLY, no model inference.

does a learned score that uses ONLY
NF4-available signals (no auxiliary INT8 pass) beat the decision margin, and is the gain stable
across patient-disjoint re-splits?

Three service scores, each under the IDENTICAL class-conditional exact split-sample certificate:
  A  margin only                      direction serve when margin >= tau   (CERTA baseline)
  B  NF4-only learned [m4, task, loglen]        logistic r-hat, serve when r-hat <= tau
  C  full learned    [m4, m8, 1{d4!=d8}, task, loglen]   logistic r-hat, serve when r-hat <= tau

Here m8/d8 are the bitsandbytes LLM.int8() margin/decision (keys in cert_<model>_bnb.npz). B needs only
the NF4 forward pass that CERTA already runs. C additionally needs the LLM.int8() margin m8 and the
NF4/LLM.int8() agreement flag, i.e. a second quantized pass at serve time. The LLM.int8() route produces
degenerate outputs on two BF16-native 70B checkpoints (Med42-70B, Llama-3-70B); the full score C is
therefore defined and reported only on the nine models with a usable LLM.int8() pass, while the single-pass
score B is reported for all eleven.

Logistic regression: scikit-learn LogisticRegression (L2 penalty, default C=1.0), features standardized by
StandardScaler fit on the selection subset, class imbalance handled by class_weight='balanced', one
predictor fit per model on the selection subset only; the predicted disagreement probability is the service
score and the serve threshold tau is chosen by the same certificate grid as the margin.

Uses only recorded data: cert_<model>_bnb.npz (ref_d,d4,d8,m4,m8) + prompts.jsonl (task,uid,user).
Requires: numpy, scipy, scikit-learn.
"""
import os, sys, json, numpy as np
from collections import defaultdict
from scipy.special import betaincinv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

CALFRAC, SELFRAC, G, DELTA = 0.5, 0.20, 50, 0.10
ORDER = ["qwen2.5-72b", "huatuogpt-o1-72b", "med42-70b", "openbiollm-70b", "mmed-3.1-70b", "llama3.3-70b",
         "llama3-70b-base", "llama3-8b", "mmed-8b", "mistral-7b", "biomistral-7b"]
DISP = dict(zip(ORDER, ["Qwen2.5-72B", "HuatuoGPT-o1-72B", "Med42-70B", "OpenBioLLM-70B",
                        "MMed-Llama-3.1-70B", "Llama-3.3-70B", "Llama-3-70B", "Llama-3-8B",
                        "MMed-Llama-3-8B", "Mistral-7B", "BioMistral-7B"]))
DIR = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CERTA_VERIFY_DIR", "data/verify")).rstrip("/") + "/"
rows = [json.loads(l) for l in open(DIR + "prompts.jsonl") if l.strip()]
UID = np.array([str(r["uid"]) for r in rows])
TASK = np.array([1.0 if r["task"] == "qar" else 0.0 for r in rows])
LOGLEN = np.array([np.log1p(r["nwords"] if "nwords" in r else len(r["user"].split())) for r in rows])
N = len(rows)

def pgroup(idx, s, f):
    g = defaultdict(list)
    for j in idx: g[UID[j]].append(j)
    gk = list(g); np.random.default_rng(s).shuffle(gk); t = int(round(f * len(idx))); A = []
    for k in gk:
        if len(A) < t: A += g[k]
    S = set(A); return np.array(sorted(S)), np.array([j for j in idx if j not in S])

def one_per_patient(idx, s):
    g = defaultdict(list)
    for j in idx: g[UID[j]].append(j)
    rng = np.random.default_rng(s + 7); return np.array(sorted(int(rng.choice(v)) for v in g.values()))

def cp_upper(x, n, d):
    return 1.0 if (n == 0 or x >= n) else float(betaincinv(x + 1, n - x, 1.0 - d))

def mond_served(score, direction, d4, fl, alpha, idx, sel, cpp):
    """Class-conditional (Mondrian) exact split-sample certificate; served mask over idx."""
    dh = DELTA / 2.0; served = np.zeros(len(idx), bool)
    for g in (0, 1):
        selg = sel[d4[sel] == g]; certg = cpp[d4[cpp] == g]
        if len(selg) == 0 or len(certg) == 0:
            continue
        cand = np.unique(np.quantile(score[selg], np.linspace(0, 1, G))); dp = dh / len(cand); best = None
        for t in cand:
            sv = certg[score[certg] >= t] if direction == 'ge' else certg[score[certg] <= t]
            if cp_upper(int(fl[sv].sum()), len(sv), dp) <= alpha:
                best = t if best is None else (min(best, t) if direction == 'ge' else max(best, t))
        if best is None:
            continue
        cond = (score[idx] >= best) if direction == 'ge' else (score[idx] <= best)
        served |= ((d4[idx] == g) & cond)
    return served

FEAT = {
    'B': lambda m4, m8, d4, d8: np.column_stack([m4, TASK, LOGLEN]),
    'C': lambda m4, m8, d4, d8: np.column_stack([m4, m8, (d4 != d8).astype(float), TASK, LOGLEN]),
}

def run_seed(alpha, seed):
    cal, test = pgroup(np.arange(N), seed, CALFRAC)
    sel, _cert = pgroup(cal, seed + 1, SELFRAC)
    cpp = one_per_patient(_cert, seed)
    cov = {k: [] for k in ('A', 'B', 'C')}; dis = {k: [] for k in ('A', 'B', 'C')}
    for m in ORDER:
        z = np.load(f"{DIR}cert_{m}_bnb.npz")
        ref, d4, d8 = z["ref_d"], z["d4"], z["d8"]; m4, m8 = z["m4"].astype(float), z["m8"].astype(float)
        fl = (d4 != ref).astype(int)
        sA = mond_served(m4, 'ge', d4, fl, alpha, test, sel, cpp)
        cov['A'].append(100 * sA.mean()); dis['A'].append(100 * fl[test][sA].mean() if sA.sum() else 0.0)
        ys = fl[sel]
        for key in ('B', 'C'):
            if 3 <= ys.sum() < len(ys):
                X = FEAT[key](m4, m8, d4, d8); Xs = StandardScaler().fit(X[sel]).transform(X)
                r = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Xs[sel], ys).predict_proba(Xs)[:, 1]
                sv = mond_served(r, 'le', d4, fl, alpha, test, sel, cpp)
            else:
                sv = sA
            cov[key].append(100 * sv.mean()); dis[key].append(100 * fl[test][sv].mean() if sv.sum() else 0.0)
    return {k: np.array(v) for k, v in cov.items()}, {k: np.array(v) for k, v in dis.items()}

def boot_ci(d, s=1, B=2000):
    rng = np.random.default_rng(s)
    bs = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])

# The full (two-pass) score uses the LLM.int8() margin and the NF4/LLM.int8() agreement. The LLM.int8()
# route produces degenerate outputs on two BF16-native 70B checkpoints (Med42-70B, Llama-3-70B); those
# configurations have no usable 8-bit pass, so the full score is defined and reported only on the nine
# models with a working LLM.int8() route. The NF4-only score (single pass) is reported for all eleven.
FAILED_INT8 = ("med42-70b", "llama3-70b-base")
valid = np.array([m not in FAILED_INT8 for m in ORDER])

# ---- primary endpoint: alpha = 5%, seed 0 ----
print(f"data dir: {DIR}   models: {len(ORDER)} (full/two-pass score on {int(valid.sum())} with usable LLM.int8())\n")
cov, dis = run_seed(0.05, 0)
A, Bc, Cc = cov['A'], cov['B'], cov['C']
dB, dC = Bc - A, Cc - A
print("=== alpha=5% class-conditional certified coverage (seed 0) ===")
print(f"{'Model':20} {'margin':>7} {'NF4-only':>9} {'full':>7} {'dNF4':>6} {'dfull':>6}")
for i, m in enumerate(ORDER):
    cstr = f"{Cc[i]:7.1f}" if valid[i] else f"{'n/a':>7}"
    dstr = f"{dC[i]:+6.1f}" if valid[i] else f"{'--':>6}"
    print(f"{DISP[m]:20} {A[i]:7.1f} {Bc[i]:9.1f} {cstr} {dB[i]:+6.1f} {dstr}")
print(f"{'Mean (all 11)':20} {A.mean():7.1f} {Bc.mean():9.1f} {'':>7} {dB.mean():+6.1f} {'':>6}")
print(f"{'Mean (9 int8)':20} {A[valid].mean():7.1f} {Bc[valid].mean():9.1f} {Cc[valid].mean():7.1f} {dB[valid].mean():+6.1f} {dC[valid].mean():+6.1f}")
loB, hiB = boot_ci(dB); loC, hiC = boot_ci(dC[valid])
print(f"\nNF4-only vs margin (11 models): mean {dB.mean():+.1f} pp, 95% CI [{loB:+.1f},{hiB:+.1f}], improved {int((dB>0).sum())}/11")
print(f"full     vs margin (9 int8):    mean {dC[valid].mean():+.1f} pp, 95% CI [{loC:+.1f},{hiC:+.1f}], improved {int((dC[valid]>0).sum())}/9")
print("validity (held-out served-set disagreement, target 5%):")
print(f"   margin   mean {dis['A'].mean():.2f}% max {dis['A'].max():.2f}% | NF4-only mean {dis['B'].mean():.2f}% max {dis['B'].max():.2f}% | "
      f"full(9) mean {dis['C'][valid].mean():.2f}% max {dis['C'][valid].max():.2f}% | any>5%: "
      f"{int((dis['A']>5).sum()+(dis['B']>5).sum()+(dis['C'][valid]>5).sum())}")

# ---- stability across 20 patient-disjoint re-splits ----
print("\n=== stability over 20 patient-disjoint re-splits (alpha=5%) ===")
splitC = np.zeros((20, 11)); mB, mC = [], []
for s in range(20):
    c, _ = run_seed(0.05, s)
    db = c['B'] - c['A']; dc = c['C'] - c['A']; splitC[s] = dc
    mB.append(db.mean()); mC.append(dc[valid].mean())
mB, mC = np.array(mB), np.array(mC)
print(f"NF4-only (11): split-mean {mB.mean():+.1f} pp, sd {mB.std():.1f}, range [{mB.min():+.1f},{mB.max():+.1f}], splits>0: {int((mB>0).sum())}/20")
print(f"full (9 int8): split-mean {mC.mean():+.1f} pp, sd {mC.std():.1f}, range [{mC.min():+.1f},{mC.max():+.1f}], splits>0: {int((mC>0).sum())}/20")
print("per-model full-score change over 20 splits  (median [IQR]):")
for i, m in enumerate(ORDER):
    if valid[i]:
        col = splitC[:, i]; q1, md, q3 = np.percentile(col, [25, 50, 75])
        print(f"   {DISP[m]:20} {md:+.1f} [{q1:+.1f}, {q3:+.1f}]")

# ---- stricter level ----
cov1, _ = run_seed(0.01, 0)
A1, B1, C1 = cov1['A'], cov1['B'], cov1['C']
print(f"\nalpha=1% (seed 0): margin {A1.mean():.1f}%  NF4-only {B1.mean():.1f}%  full(9) {C1[valid].mean():.1f}%  "
      f"(dNF4 {(B1-A1).mean():+.1f}/11, dfull {(C1[valid]-A1[valid]).mean():+.1f}/9)")
