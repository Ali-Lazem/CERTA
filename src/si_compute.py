#!/usr/bin/env python3
"""Compute every SI quantity under the split-sample certificate (primary: grid from the selection
subset, CP from the certification subset, coverage on test) and the full-calibration certificate
(sensitivity). Saves si_data.pkl. Uses certa_core as the single source of truth."""
import numpy as np, pickle, csv
from collections import defaultdict
from certa_core import (DIR, ORDER, CONF, LAB, TASK, N, A, DELTA, GRID, arr, cp_upper,
                        marg, marg_frontier, mond, mond_frontier, _tau,
                        SP, FULL, SPLIT, SEL, CERT, CAL, TEST, make_split, pgroup_split)
taskT=np.array([TASK[i] for i in TEST])
def tauM(mg,fl,sp=SP):                     # marginal split threshold
    return _tau(mg,fl,sp["sel"],sp["cert"],A,DELTA)
def tauC(mg,fl,d,sp=SP):                    # per-class split thresholds (delta/2)
    return {c:_tau(mg,fl,sp["sel"][d[sp["sel"]]==c],sp["cert"][d[sp["cert"]]==c],A,DELTA/2) for c in (0,1)}

D={}
D["perconfig"]={m:{LAB[(q,b)]:marg(m,q,b) for q in CONF for b in CONF[q]} for m in ORDER}
dd={}
for m in ORDER:
    ref,d,_=arr(m,"bnb",4); row=dict(refNO=100*ref.mean(),qNO=100*d.mean())
    for tk in ("ner","qar"):
        idx=np.array([i for i in range(N) if TASK[i]==tk]); row[f"ref_{tk}"]=100*ref[idx].mean(); row[f"q_{tk}"]=100*d[idx].mean()
    dd[m]=row
D["decdist"]=dd
D["audit"]={m:marg(m,"bnb",4) for m in ORDER}
D["mond"]={m:mond_frontier(m) for m in ORDER}                    # primary (full calibration)
D["mond_split"]={m:mond_frontier(m,sp=SPLIT) for m in ORDER}     # split-sample exact-guarantee sensitivity
D["marg_split"]={m:marg_frontier(m,sp=SPLIT)[2] for m in ORDER}
D["marg_primary"]={m:marg_frontier(m)[2] for m in ORDER}
D["delta"]={m:{dl:marg_frontier(m,delta=dl)[2]["size"] for dl in (0.10,0.05,0.01)} for m in ORDER}
cs={}
for nc in (250,500,750,1000,1250):
    cal,test=pgroup_split(np.arange(N),0,target=nc); sp=dict(cal=cal,test=test,sel=cal,cert=cal)
    sz=[marg_frontier(m,sp=sp)[2]["size"] for m in ORDER]
    cs[nc]=dict(lo=min(sz),hi=max(sz),med=sorted(sz)[len(sz)//2])
D["calsize"]=cs
tw={}
for q in CONF:
    for b in CONF[q]:
        cN=[];cQ=[];dN=[];dQ=[]
        for m in ORDER:
            ref,d,mg=arr(m,q,b); fl=(d!=ref).astype(int); tau=tauM(mg,fl)
            for tk,cl,dl in [("ner",cN,dN),("qar",cQ,dQ)]:
                s=TEST[taskT==tk]; cov=mg[s]>=tau
                cl.append(100*cov.mean()); dl.append(100*fl[s][cov].mean() if cov.sum() else 0.0)
        tw[LAB[(q,b)]]=dict(nerCov=np.mean(cN),nerDis=np.mean(dN),qaCov=np.mean(cQ),qaDis=np.mean(dQ))
D["taskwise"]=tw
esc={}
for m in ORDER:
    ref,d,mg=arr(m,"bnb",4); fl=(d!=ref).astype(int); tau=tauM(mg,fl); cov=(mg[TEST]>=tau)
    esc[m]=dict(mCert=float(mg[TEST][cov].mean()),mEsc=float(mg[TEST][~cov].mean()) if (~cov).sum() else float('nan'),
                disCert=100*fl[TEST][cov].mean() if cov.sum() else 0.0,
                disEsc=100*fl[TEST][~cov].mean() if (~cov).sum() else float('nan'),escFrac=100*(~cov).mean())
D["escalated"]=esc
QS=[0.5,0.4,0.3,0.2,0.1,0.05]; prev={"marg":defaultdict(list),"mond":defaultdict(list)}; per_model_prev={}
for m in ORDER:
    ref,d,mg=arr(m,"bnb",4); fl=(d!=ref).astype(int); tM=tauM(mg,fl); tC=tauC(mg,fl,d)
    stat={}
    for name,tc in [("marg",{0:tM,1:tM}),("mond",tC)]:
        s={}
        for c in (0,1):
            tt=TEST[d[TEST]==c]; se=mg[tt]>=tc[c]; s[c]=(se.mean() if len(tt) else 0.0,(fl[tt][se].mean() if se.sum() else 0.0))
        stat[name]=s
    pm={}
    for name in ("marg","mond"):
        s=stat[name]; series=[]
        for q in QS:
            wY,wN=1-q,q; mY=wY*s[0][0]; mN=wN*s[1][0]; dis=100*(mY*s[0][1]+mN*s[1][1])/(mY+mN) if (mY+mN)>0 else 0.0
            series.append(dis); prev[name][q].append(dis)
        pm[name]=series
    per_model_prev[m]=pm
D["prev_qs"]=QS; D["prev_agg"]={name:{q:float(np.mean(prev[name][q])) for q in QS} for name in ("marg","mond")}; D["prev_model"]=per_model_prev
ner=defaultdict(dict)
for r in csv.DictReader(open(DIR+"bench.csv")):
    ner[(r["model"],r["dataset"])][(r["quantizer"],r["config"])]=(float(r["precision"]),float(r["recall"]),float(r["f1"]))
D["ner"]={k:v for k,v in ner.items()}
pickle.dump(D,open("si_data.pkl","wb"))
print("saved si_data.pkl (full-calibration primary; split-sample sensitivity). cal=%d test=%d"%(len(CAL),len(TEST)))
for m in ORDER:
    mo=D["mond"][m]; print("  %-16s primary mond %.1f esc %.1f cfg %s-%d | split %.1f"%(m,mo["size"],mo["esc"],mo["q"],mo["bit"],D["mond_split"][m]["size"]))
