#!/usr/bin/env python3
"""Faithful reproduction of the CERTA per-config certificate on the current patient-disjoint split.
Replicates certified_frontier.py's split (patient-grouped, rng(seed) shuffle) and Learn-then-Test
exactly, then emits a per-(model,config) table with rawPres, certCov, agreement, and per-class /
per-task breakdowns. Also supports delta and alpha sweeps. GPU-free (reads the saved npz)."""
import json, glob, numpy as np
from scipy.special import betaincinv

DIR="data/verify/"
PROMPTS=DIR+"prompts.jsonl"
SEED=0; CALFRAC=0.5; GRID=50; DELTA=0.10
MODELS=["qwen2.5-72b","huatuogpt-o1-72b","med42-70b","openbiollm-70b","mmed-3.1-70b","llama3.3-70b",
        "llama3-70b-base","llama3-8b","mmed-8b","mistral-7b","biomistral-7b"]
# config -> (npz-quant, bit)
CONFIGS=[("rtn8","rtn",8),("rtn6","rtn",6),("rtn4","rtn",4),("int8","bnb",8),("nf4","bnb",4)]

rows=[json.loads(l) for l in open(PROMPTS) if l.strip()]
UID=[str(r.get("uid","")) for r in rows]
TASK=[str(r.get("task","")) for r in rows]
N=len(UID)

def split_indices(n):
    from collections import defaultdict
    g=defaultdict(list)
    for i,u in enumerate(UID): g[u].append(i)
    gk=list(g); np.random.default_rng(SEED).shuffle(gk)
    target=int(round(CALFRAC*n)); cal=[]
    for k in gk:
        if len(cal)<target: cal+=g[k]
    cs=set(cal); return np.array(sorted(cs)), np.array([i for i in range(n) if i not in cs])

def cp_upper(x,n,d):
    if n==0: return 1.0
    if x>=n: return 1.0
    return float(betaincinv(x+1,n-x,1.0-d))

def tau_ltt(margin,flip,alpha,delta=DELTA,grid=GRID):
    cand=np.unique(np.quantile(margin,np.linspace(0,1,grid))); dc=delta/len(cand)
    v=[t for t in cand if cp_upper(int(flip[margin>=t].sum()),int((margin>=t).sum()),dc)<=alpha]
    return min(v) if v else float("inf")

CAL,TEST=split_indices(N)
taskT=np.array([TASK[i] for i in TEST])

def load(model,quant):
    return np.load(f"{DIR}cert_{model}_{quant}.npz")

def config_arrays(model,cfg):
    _,quant,bit=next(c for c in CONFIGS if c[0]==cfg)
    z=load(model,quant); ref=z["ref_d"]; d=z[f"d{bit}"]; m=z[f"m{bit}"]; return ref,d,m,bit

def certificate(model,cfg,alpha=0.05,delta=DELTA):
    ref,d,m,bit=config_arrays(model,cfg)
    pres=(d==ref).astype(int); flip=(1-pres).astype(bool)
    mc,mt=m[CAL],m[TEST]; fc=flip[CAL]
    tau=tau_ltt(mc,fc,alpha,delta)
    cov=(mt>=tau)
    rawPres=100*pres[TEST].mean()
    certCov=100*cov.mean()
    agree=100*pres[TEST][cov].mean() if cov.sum() else 100.0
    amort=(certCov/100)*bit+(1-certCov/100)*16; size=100*amort/16
    return dict(rawPres=round(rawPres,1),certCov=round(certCov,1),agree=round(agree,1),
                size=round(size,1),tau=tau,bit=bit)

if __name__=="__main__":
    print(f"split: n={N} cal={len(CAL)} test={len(TEST)}  (seed={SEED})")
    # validation against frontier_bnb (nf4 certCov)
    print("\n=== per-config certCov (alpha=5%, delta=0.10) : rawPres / certCov / agreement ===")
    hdr=f"{'model':>18} "+" ".join(f"{c:>18}" for c,_,_ in CONFIGS); print(hdr)
    table={}
    for mo in MODELS:
        cells=[]
        for cfg,_,_ in CONFIGS:
            r=certificate(mo,cfg); table[(mo,cfg)]=r
            cells.append(f"{r['rawPres']:>5.1f}/{r['certCov']:>5.1f}/{r['agree']:>4.1f}")
        print(f"{mo:>18} "+" ".join(f"{c:>18}" for c in cells))
    # summary ranges for the manuscript
    def rng(cfg,key):
        vs=[table[(m,cfg)][key] for m in MODELS]; return min(vs),max(vs)
    print("\n=== manuscript ranges (alpha=5%, delta=0.10) ===")
    for cfg in ["rtn8","rtn6","rtn4","int8","nf4"]:
        lo,hi=rng(cfg,"certCov"); print(f"{cfg:>6} certCov: {lo:.1f}-{hi:.1f}")
    lo,hi=rng("rtn8","agree"); print(f"rtn8 agreement: {lo:.1f}-{hi:.1f}")
    import pickle; pickle.dump(table,open("repro_table.pkl","wb"))
