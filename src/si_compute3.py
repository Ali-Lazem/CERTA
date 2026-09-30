#!/usr/bin/env python3
"""SI-v3 additions under the split-sample certificate: escalation mechanism (resolution- vs
risk-limited) on the certification subset, threshold-grid sensitivity, repeated patient-disjoint split
stability, class-conditional delta- and alpha-sensitivity. Saves si_add3.pkl. Uses certa_core."""
import numpy as np, pickle
from certa_core import (ORDER, DELTA, A, GRID, arr, cp_upper, _tau, CAL, pgroup_split, N,
                        marg_frontier, mond_frontier)
ADD={}
# (1) reject-class escalation mechanism + min attainable CP (full calibration, NF4)
reason={}; mincp={}
for m in ORDER:
    ref,d,mg=arr(m,"bnb",4); fl=(d!=ref).astype(int); dh=DELTA/2
    ci=CAL[d[CAL]==1]; nN=len(ci)
    floorN=100*(1-(dh/GRID)**(1/max(nN,1)))
    cand=np.unique(np.quantile(mg[ci],np.linspace(0,1,GRID))) if nN else np.array([])
    dc=dh/max(len(cand),1)
    cps=[100*cp_upper(int(fl[ci][mg[ci]>=t].sum()),int((mg[ci]>=t).sum()),dc) for t in cand] if nN else [100.0]
    mincp[m]=round(min(cps) if cps else 100.0,1)
    tauN=_tau(mg,fl,ci,ci,A,dh)
    reason[m]="certified" if np.isfinite(tauN) else ("resolution-limited" if floorN>A*100 else "risk-limited")
ADD["reason"]=reason; ADD["mincpN"]=mincp
# (2) threshold-grid sensitivity (vary G)
grids=[25,50,100,200]; ADD["grid"]={}
for G in grids:
    fr={m:marg_frontier(m,grid=G)[2]["size"] for m in ORDER}; sz=list(fr.values())
    ADD["grid"][G]=dict(lo=round(min(sz),1),hi=round(max(sz),1),med=round(sorted(sz)[5],1),
                        flag={m:round(fr[m],1) for m in ["qwen2.5-72b","huatuogpt-o1-72b","llama3.3-70b"]})
# (3) repeated patient-disjoint split stability (20 splits; full-calibration certificate on each)
splitfr={m:[] for m in ORDER}
for s in range(20):
    cal,test=pgroup_split(np.arange(N),s,frac=0.5); sp=dict(cal=cal,test=test,sel=cal,cert=cal)
    for m in ORDER: splitfr[m].append(marg_frontier(m,sp=sp)[2]["size"])
ADD["split"]={m:dict(med=round(float(np.median(v)),1),iqr=tuple(round(float(x),1) for x in np.percentile(v,[25,75])),
                     lo=round(min(v),1),hi=round(max(v),1)) for m,v in splitfr.items()}
ADD["split_raw"]=splitfr
# (4) class-conditional delta-sensitivity
ADD["monddelta"]={m:{dl:round(mond_frontier(m,delta_joint=dl)["size"],1) for dl in (0.10,0.05,0.01)} for m in ORDER}
# (5) class-conditional alpha=1/5/10 frontier (size + escalation)
ADD["mondalpha"]={}
for m in ORDER:
    dd={}
    for a in (0.01,0.05,0.10):
        fr=mond_frontier(m,alpha=a); dd[a]={"size":round(fr["size"],1),"esc":round(fr["esc"],1)}
    ADD["mondalpha"][m]=dd
pickle.dump(ADD,open("si_add3.pkl","wb"))
print("saved si_add3.pkl (split-primary)")
for m in ORDER: print("  %-16s %-18s minCP %.1f"%(m,ADD["reason"][m],ADD["mincpN"][m]))
print("grid range:",{G:(ADD["grid"][G]["lo"],ADD["grid"][G]["hi"]) for G in grids})
iqrw=max(ADD["split"][m]["iqr"][1]-ADD["split"][m]["iqr"][0] for m in ORDER)
print("max split IQR width: %.1f"%iqrw)
