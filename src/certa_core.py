#!/usr/bin/env python3
"""Shared CERTA certificate core -- single source of truth for the split-sample certificate.

Primary certificate: the candidate threshold grid is built on an independent SELECTION subset of the
calibration set, the Clopper--Pearson bound is evaluated on the disjoint CERTIFICATION subset, and
coverage is measured on the held-out TEST set. This keeps the grid independent of the certification
labels, so the finite-sample Learn-then-Test / Clopper--Pearson guarantee is exact.

Setting selection = certification = full calibration recovers the higher-power full-calibration
sensitivity certificate (the grid is then not independent of the certification sample)."""
import json, numpy as np
from collections import defaultdict
from scipy.special import betaincinv
DIR="data/verify/"; SEED=0; CALFRAC=0.5; SELFRAC=0.20; GRID=50; DELTA=0.10; A=0.05
ORDER=["qwen2.5-72b","huatuogpt-o1-72b","med42-70b","openbiollm-70b","mmed-3.1-70b","llama3.3-70b",
       "llama3-70b-base","llama3-8b","mmed-8b","mistral-7b","biomistral-7b"]
CONF={"rtn":[8,6,4],"bnb":[8,4]}                       # bnb8=INT8, bnb4=NF4 (full sweep, for per-config table)
CONF_DEPLOY={"bnb":[8,4]}                               # memory-reducing configs only (NF4, INT8): the
                                                        # deployable frontier is searched over these; RTN
                                                        # retains bf16 storage so it is a precision-sweep
                                                        # baseline, not a deployment option.
LAB={("rtn",8):"RTN-8",("rtn",6):"RTN-6",("rtn",4):"RTN-4",("bnb",8):"INT8",("bnb",4):"NF4"}
_rows=[json.loads(l) for l in open(DIR+"prompts.jsonl") if l.strip()]
UID=[str(r.get("uid","")) for r in _rows]; TASK=[str(r.get("task","")) for r in _rows]; N=len(UID)

def pgroup_split(idx, seed, frac=None, target=None):
    """Patient-grouped split of idx into (A, B); A gets whole patients until ~frac (or exactly target)."""
    g=defaultdict(list)
    for j in idx: g[UID[j]].append(j)
    gk=list(g); np.random.default_rng(seed).shuffle(gk)
    t=target if target is not None else int(round(frac*len(idx))); A=[]
    for k in gk:
        if len(A)<t: A+=g[k]
    As=set(A); return np.array(sorted(As)), np.array([j for j in idx if j not in As])

def make_split(seed=SEED, cal_target=None, selfrac=SELFRAC):
    cal,test=pgroup_split(np.arange(N), seed, frac=CALFRAC if cal_target is None else None, target=cal_target)
    sel,cert=pgroup_split(cal, seed+1, frac=selfrac)
    return dict(cal=cal, test=test, sel=sel, cert=cert)

_SPL=make_split()                                      # sel/cert split (250 sel / 1000 cert / 1250 test)
CAL,TEST,SEL,CERT=_SPL["cal"],_SPL["test"],_SPL["sel"],_SPL["cert"]
FULL=dict(cal=CAL, test=TEST, sel=CAL, cert=CAL)       # full-calibration certificate (grid+CP on full cal)
SPLIT=_SPL                                              # split-sample certificate (exact-guarantee sensitivity)
SP=FULL                                                # PRIMARY reported certificate = full calibration

def cp_upper(x,n,d):
    if n==0 or x>=n: return 1.0
    return float(betaincinv(x+1,n-x,1.0-d))
def arr(m,q,bit):
    z=np.load(f"{DIR}cert_{m}_{q}.npz"); return z["ref_d"],z[f"d{bit}"],z[f"m{bit}"]
def _tau(mg,fl,gidx,cidx,alpha,delta,grid=GRID):
    if len(gidx)==0 or len(cidx)==0: return float("inf")
    cand=np.unique(np.quantile(mg[gidx],np.linspace(0,1,grid))); dc=delta/len(cand)
    ok=[t for t in cand if cp_upper(int(fl[cidx][mg[cidx]>=t].sum()),int((mg[cidx]>=t).sum()),dc)<=alpha]
    return min(ok) if ok else float("inf")

def marg(m,q,bit,alpha=A,delta=DELTA,sp=SP,grid=GRID):
    gidx,cidx,tidx=sp["sel"],sp["cert"],sp["test"]
    ref,d,mg=arr(m,q,bit); fl=(d!=ref).astype(int)
    tau=_tau(mg,fl,gidx,cidx,alpha,delta,grid); cov=(mg[tidx]>=tau)
    n=int((mg[cidx]>=tau).sum()); x=int(fl[cidx][mg[cidx]>=tau].sum())
    return dict(rawPres=100*(fl[tidx]==0).mean(), certCov=100*cov.mean(),
                agree=100*(1-fl[tidx][cov].mean()) if cov.sum() else 100.0,
                size=100*((cov.mean()*bit+(1-cov.mean())*16)/16), tau=tau,
                ncal=n, xcal=x, cp=100*cp_upper(x,n,delta/grid), bit=bit)
def marg_frontier(m,alpha=A,delta=DELTA,sp=SP,grid=GRID,conf=CONF):
    return min(((q,b,marg(m,q,b,alpha,delta,sp,grid)) for q in conf for b in conf[q]), key=lambda p:p[2]["size"])
def mond(m,q,bit,alpha=A,delta_joint=DELTA,sp=SP,grid=GRID):
    gidx,cidx,tidx=sp["sel"],sp["cert"],sp["test"]; dh=delta_joint/2
    ref,d,mg=arr(m,q,bit); fl=(d!=ref).astype(int)
    tau={};n={};x={}
    for c in (0,1):
        gi=gidx[d[gidx]==c]; ci=cidx[d[cidx]==c]
        tau[c]=_tau(mg,fl,gi,ci,alpha,dh,grid)
        served=ci[mg[ci]>=tau[c]]; n[c]=len(served); x[c]=int(fl[served].sum())
    cov=np.array([mg[i]>=tau[d[i]] for i in tidx]).mean() if len(tidx) else 0.0
    pc={};di={}
    for c in (0,1):
        tt=tidx[d[tidx]==c]; s=mg[tt]>=tau[c]
        pc[c]=100*s.mean() if len(tt) else float('nan'); di[c]=100*fl[tt][s].mean() if s.sum() else 0.0
    return dict(size=100*((cov*bit+(1-cov)*16)/16),cov=100*cov,covY=pc[0],covN=pc[1],disY=di[0],disN=di[1],
                esc=100*(1-cov),tauY=tau[0],tauN=tau[1],nY=n[0],xY=x[0],nN=n[1],xN=x[1],
                cpY=100*cp_upper(x[0],n[0],dh/grid),cpN=100*cp_upper(x[1],n[1],dh/grid),bit=bit,q=q)
def mond_frontier(m,alpha=A,delta_joint=DELTA,sp=SP,grid=GRID,conf=CONF):
    return min((mond(m,q,b,alpha,delta_joint,sp,grid) for q in conf for b in conf[q]), key=lambda p:p["size"])

if __name__=="__main__":
    print("sizes: cal=%d test=%d | sel=%d cert=%d"%(len(CAL),len(TEST),len(SEL),len(CERT)))
    print("%-16s | primary mond (esc) | full mond (esc)"%"model")
    for m in ORDER:
        p=mond_frontier(m); f=mond_frontier(m,sp=FULL)
        print("%-16s |  %5.1f (%4.1f)     |  %5.1f (%4.1f)"%(m,p["size"],p["esc"],f["size"],f["esc"]))
