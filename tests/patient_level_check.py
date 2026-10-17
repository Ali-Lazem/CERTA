#!/usr/bin/env python3
"""Priority-1 check: does one-decision-per-patient (2,484) change the frontier vs full (2,500)?
Reports marginal frontier and class-conditional (Mondrian) size+escalation at alpha=5%, for both."""
import json, numpy as np
from collections import defaultdict
from scipy.special import betaincinv
DIR="data/verify/"; DELTA=0.10; G=50; A=0.05
ORDER=["qwen2.5-72b","huatuogpt-o1-72b","med42-70b","openbiollm-70b","mmed-3.1-70b","llama3.3-70b",
"llama3-70b-base","llama3-8b","mmed-8b","mistral-7b","biomistral-7b"]
CONF={"bnb":[8,4]}   # deployable (memory-reducing) configs only: INT8, NF4
rows=[json.loads(l) for l in open(DIR+"prompts.jsonl") if l.strip()]
UID=[str(r.get("uid","")) for r in rows]; N=len(UID)
seen=set(); opp=[]
for i,u in enumerate(UID):
    if u not in seen: seen.add(u); opp.append(i)
opp=np.array(opp)
print(f"full N={N}  patients={len(set(UID))}  one-per-patient N={len(opp)}  dropped={N-len(opp)}")

def cp_upper(x,n,d):
    if n==0 or x>=n: return 1.0
    return float(betaincinv(x+1,n-x,1.0-d))
def tau_ltt(mg,fl,alpha,delta,grid=G):
    if len(mg)==0: return float("inf")
    cand=np.unique(np.quantile(mg,np.linspace(0,1,grid))); dc=delta/len(cand)
    v=[t for t in cand if cp_upper(int(fl[mg>=t].sum()),int((mg>=t).sum()),dc)<=alpha]
    return min(v) if v else float("inf")
def arr(m,q,b):
    z=np.load(f"{DIR}cert_{m}_{q}.npz"); return z["ref_d"],z[f"d{b}"],z[f"m{b}"]
def split_subset(idx,seed):
    g=defaultdict(list)
    for j in idx: g[UID[j]].append(j)
    gk=list(g); np.random.default_rng(seed).shuffle(gk)
    t=int(round(0.5*len(idx))); cal=[]
    for k in gk:
        if len(cal)<t: cal+=g[k]
    cs=set(cal); return np.array(sorted(cs)),np.array([j for j in idx if j not in cs])
def marg_size(m,q,b,cal,test):
    ref,d,mg=arr(m,q,b); fl=(d!=ref).astype(int)
    tau=tau_ltt(mg[cal],fl[cal],A,DELTA); cov=(mg[test]>=tau).mean()
    return 100*((cov*b+(1-cov)*16)/16)
def marg_frontier(m,cal,test):
    return min(marg_size(m,q,b,cal,test) for q in CONF for b in CONF[q])
def mond(m,q,b,cal,test):
    ref,d,mg=arr(m,q,b); fl=(d!=ref).astype(int); dh=DELTA/2
    tau={c:tau_ltt(mg[cal[d[cal]==c]],fl[cal[d[cal]==c]],A,dh) for c in (0,1)}
    cov=np.array([mg[i]>=tau[d[i]] for i in test]).mean()
    return dict(size=100*((cov*b+(1-cov)*16)/16),esc=100*(1-cov))
def mond_frontier(m,cal,test):
    return min((mond(m,q,b,cal,test) for q in CONF for b in CONF[q]),key=lambda p:p["size"])

allidx=np.arange(N)
cf,tf=split_subset(allidx,0); co,to=split_subset(opp,0)
print(f"{'model':>16} | marg full | marg opp |  mond full (esc) |  mond opp (esc)  | dSize")
maxd=0
for m in ORDER:
    fm=marg_frontier(m,cf,tf); om=marg_frontier(m,co,to)
    fo=mond_frontier(m,cf,tf); oo=mond_frontier(m,co,to)
    d=abs(fo['size']-oo['size']); maxd=max(maxd,d,abs(fm-om))
    print(f"{m:>16} | {fm:6.1f}   | {om:6.1f}  |  {fo['size']:5.1f} ({fo['esc']:4.1f})  |  {oo['size']:5.1f} ({oo['esc']:4.1f})  | {d:+.1f}")
print(f"\nmax |Delta size| across models (marg or mond) = {maxd:.1f} pp BF16")
