#!/usr/bin/env python3
"""SI additions under the split-sample certificate: patient-clustered bootstrap 95% CIs (over test),
class-specific calibration resolution on the certification subset, all-11 prevalence maxima, pooled
certified/escalated margins. Saves si_add.pkl. Uses certa_core."""
import numpy as np, pickle
from collections import defaultdict
from certa_core import (ORDER, CONF, LAB, UID, DELTA, A, GRID, arr, cp_upper, _tau,
                        CAL, TEST)
NBOOT=1000
tg=defaultdict(list)
for i in TEST: tg[UID[i]].append(i)
TESTPAT=list(tg.values()); rng=np.random.default_rng(1)
def tauM(mg,fl): return _tau(mg,fl,CAL,CAL,A,DELTA)                       # full-calibration primary
def tauCls(mg,fl,d,c): return _tau(mg,fl,CAL[d[CAL]==c],CAL[d[CAL]==c],A,DELTA/2)
def boot_ci(statfn):
    vals=[]; npat=len(TESTPAT)
    for _ in range(NBOOT):
        pick=rng.integers(0,npat,npat); b=np.concatenate([TESTPAT[p] for p in pick]); vals.append(statfn(b))
    return tuple(round(x,1) for x in np.percentile(vals,[2.5,97.5]))

ADD={"ci_perconfig":{},"ci_audit":{},"ci_mond":{},"classres":{},"prev_max":{}}
for m in ORDER:
    ADD["ci_perconfig"][m]={}
    for q in CONF:
        for b in CONF[q]:
            ref,d,mg=arr(m,q,b); fl=(d!=ref).astype(int); tau=tauM(mg,fl)
            cov_stat=lambda ix,mg=mg,tau=tau:100*(mg[ix]>=tau).mean()
            ADD["ci_perconfig"][m][LAB[(q,b)]]=boot_ci(cov_stat)
            if (q,b)==("bnb",4):
                dis_stat=lambda ix,mg=mg,tau=tau,fl=fl:(100*fl[ix][mg[ix]>=tau].mean() if (mg[ix]>=tau).sum() else 0.0)
                ADD["ci_audit"][m]=dict(covCI=boot_ci(cov_stat),disCI=boot_ci(dis_stat))

for m in ORDER:
    ref,d,mg=arr(m,"bnb",4); fl=(d!=ref).astype(int); dh=DELTA/2
    nY=int((d[CAL]==0).sum()); nN=int((d[CAL]==1).sum())            # full-calibration class counts
    floorY=100*(1-(dh/GRID)**(1/max(nY,1))); floorN=100*(1-(dh/GRID)**(1/max(nN,1)))
    tau={c:tauCls(mg,fl,d,c) for c in (0,1)}
    ADD["classres"][m]=dict(nY=nY,nN=nN,floorY=round(floorY,2),floorN=round(floorN,2),
                            certY=np.isfinite(tau[0]),certN=np.isfinite(tau[1]))
    ci={}
    for c in (0,1):
        tt=TEST[d[TEST]==c]
        if len(tt)==0 or not np.isfinite(tau[c]): ci[c]=(0.0,0.0); continue
        pats=defaultdict(list)
        for i in tt: pats[UID[i]].append(i)
        pg=list(pats.values()); npat=len(pg); vals=[]
        for _ in range(NBOOT):
            pick=rng.integers(0,npat,npat); b=np.concatenate([pg[p] for p in pick]); vals.append(100*(mg[b]>=tau[c]).mean())
        ci[c]=tuple(round(x,1) for x in np.percentile(vals,[2.5,97.5]))
    ADD["ci_mond"][m]=dict(covY_CI=ci[0],covN_CI=ci[1])

D=pickle.load(open("si_data.pkl","rb")); QS=D["prev_qs"]
for name in ("marg","mond"):
    ADD["prev_max"][name]={q:round(max(D["prev_model"][m][name][j] for m in ORDER),1) for j,q in enumerate(QS)}
ADD["prev_qs"]=QS
mc=[];me=[]
for m in ORDER:
    ref,d,mg=arr(m,"bnb",4); fl=(d!=ref).astype(int); tau=tauM(mg,fl); cov=(mg[TEST]>=tau)
    mc.append(mg[TEST][cov]); me.append(mg[TEST][~cov])
ADD["ecdf_cert"]=np.concatenate(mc); ADD["ecdf_esc"]=np.concatenate(me)
pickle.dump(ADD,open("si_add.pkl","wb"))
print("escalated margin median %.2f ; certified median %.2f"%(np.median(ADD["ecdf_esc"]),np.median(ADD["ecdf_cert"])))
print("classres (n_Y/n_N cert / floorN% / certN):")
for m in ORDER:
    c=ADD["classres"][m]; print(f"  {m:>16} {c['nY']:>4}/{c['nN']:>4}  floorN {c['floorN']:>6.2f}  certN={c['certN']}")
print("saved si_add.pkl")
