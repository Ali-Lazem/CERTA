#!/usr/bin/env python3
"""Impact of sample-splitting the calibration set (grid from an independent selection subset,
CP from a disjoint certification subset) vs the current single-calibration certificate.
Reports marginal and class-conditional (Mondrian) frontier at alpha=5% for both."""
import json, numpy as np
from collections import defaultdict
from scipy.special import betaincinv
DIR="data/verify/"; DELTA=0.10; G=50; A=0.05
ORDER=["qwen2.5-72b","huatuogpt-o1-72b","med42-70b","openbiollm-70b","mmed-3.1-70b","llama3.3-70b",
"llama3-70b-base","llama3-8b","mmed-8b","mistral-7b","biomistral-7b"]
CONF={"rtn":[8,6,4],"bnb":[8,4]}
rows=[json.loads(l) for l in open(DIR+"prompts.jsonl") if l.strip()]
UID=[str(r.get("uid","")) for r in rows]; N=len(UID)
def cp_upper(x,n,d):
    if n==0 or x>=n: return 1.0
    return float(betaincinv(x+1,n-x,1.0-d))
def arr(m,q,b):
    z=np.load(f"{DIR}cert_{m}_{q}.npz"); return z["ref_d"],z[f"d{b}"],z[f"m{b}"]
def pgroups(idx,seed,frac):
    """patient-grouped split of idx into (A ~ frac, B ~ rest)."""
    g=defaultdict(list)
    for j in idx: g[UID[j]].append(j)
    gk=list(g); np.random.default_rng(seed).shuffle(gk)
    t=int(round(frac*len(idx))); A=[]
    for k in gk:
        if len(A)<t: A+=g[k]
    As=set(A); return np.array(sorted(As)), np.array([j for j in idx if j not in As])
CAL,TEST=pgroups(np.arange(N),0,0.5)            # 1250 cal / 1250 test (as in the paper)
SEL,CERT=pgroups(CAL,1,0.20)                     # split cal -> 250 selection / 1000 certification
print(f"cal={len(CAL)} test={len(TEST)} | selection={len(SEL)} certification={len(CERT)}")

def grid_from(mg_idx):                            # candidate grid = deduped quantiles of given margins
    return np.unique(np.quantile(mg_idx,np.linspace(0,1,G)))

# ---- current: grid & CP both on full calibration ----
def tau_cur(mg,fl):
    cand=grid_from(mg[CAL]); dc=DELTA/len(cand)
    v=[t for t in cand if cp_upper(int(fl[CAL][mg[CAL]>=t].sum()),int((mg[CAL]>=t).sum()),dc)<=A]
    return min(v) if v else float("inf")
# ---- split: grid on SEL, CP on CERT ----
def tau_split(mg,fl):
    cand=grid_from(mg[SEL]); dc=DELTA/len(cand)
    v=[t for t in cand if cp_upper(int(fl[CERT][mg[CERT]>=t].sum()),int((mg[CERT]>=t).sum()),dc)<=A]
    return min(v) if v else float("inf")
def marg(m,taufn):
    best=1e9
    for q in CONF:
        for b in CONF[q]:
            ref,d,mg=arr(m,q,b); fl=(d!=ref).astype(int)
            tau=taufn(mg,fl); cov=(mg[TEST]>=tau).mean(); best=min(best,100*((cov*b+(1-cov)*16)/16))
    return best
def mondf(m,taufn_cls):
    best=None
    for q in CONF:
        for b in CONF[q]:
            ref,d,mg=arr(m,q,b); fl=(d!=ref).astype(int)
            tau={c:taufn_cls(mg,fl,d,c) for c in (0,1)}
            cov=np.array([mg[i]>=tau[d[i]] for i in TEST]).mean(); s=100*((cov*b+(1-cov)*16)/16)
            if best is None or s<best[0]: best=(s,100*(1-cov))
    return best
def tau_cur_cls(mg,fl,d,c):
    sub=CAL[d[CAL]==c]; cand=grid_from(mg[sub]); dc=(DELTA/2)/len(cand)
    v=[t for t in cand if cp_upper(int(fl[sub][mg[sub]>=t].sum()),int((mg[sub]>=t).sum()),dc)<=A]
    return min(v) if v else float("inf")
def tau_split_cls(mg,fl,d,c):
    s_sel=SEL[d[SEL]==c]; s_cert=CERT[d[CERT]==c]
    if len(s_sel)==0 or len(s_cert)==0: return float("inf")
    cand=grid_from(mg[s_sel]); dc=(DELTA/2)/len(cand)
    v=[t for t in cand if cp_upper(int(fl[s_cert][mg[s_cert]>=t].sum()),int((mg[s_cert]>=t).sum()),dc)<=A]
    return min(v) if v else float("inf")

print(f"{'model':>16} | marg cur | marg split |  mond cur (esc) | mond split (esc) | dMarg dMond")
mm=md=0
for m in ORDER:
    mc=marg(m,tau_cur); ms=marg(m,tau_split); oc=mondf(m,tau_cur_cls); os=mondf(m,tau_split_cls)
    mm=max(mm,abs(ms-mc)); md=max(md,abs(os[0]-oc[0]))
    print(f"{m:>16} | {mc:6.1f}   | {ms:6.1f}     | {oc[0]:5.1f} ({oc[1]:4.1f})   | {os[0]:5.1f} ({os[1]:4.1f})    | {ms-mc:+5.1f} {os[0]-oc[0]:+5.1f}")
print(f"\nmax |delta| marginal = {mm:.1f} pp ; max |delta| Mondrian = {md:.1f} pp")
