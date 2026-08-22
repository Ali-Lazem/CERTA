#!/usr/bin/env python3
"""Emit LaTeX table bodies for the SI -> si_tables.tex (\\input-ed by SI.tex). Adds patient-clustered
95% CIs and class-specific calibration resolution from si_add.pkl."""
import pickle
D=pickle.load(open("si_data.pkl","rb")); A=pickle.load(open("si_add.pkl","rb")); A3=pickle.load(open("si_add3.pkl","rb"))
ORDER=["qwen2.5-72b","huatuogpt-o1-72b","med42-70b","openbiollm-70b","mmed-3.1-70b","llama3.3-70b",
"llama3-70b-base","llama3-8b","mmed-8b","mistral-7b","biomistral-7b"]
NAME={"llama3.3-70b":"Llama-3.3-70B","llama3-70b-base":"Llama-3-70B (base)","mmed-3.1-70b":"MMed-Llama-3.1-70B",
"openbiollm-70b":"OpenBioLLM-70B","med42-70b":"Med42-70B","qwen2.5-72b":"Qwen2.5-72B",
"huatuogpt-o1-72b":"HuatuoGPT-o1-72B","llama3-8b":"Llama-3-8B","mmed-8b":"MMed-Llama-3-8B",
"mistral-7b":"Mistral-7B","biomistral-7b":"BioMistral-7B"}
def f1(x): return "%.1f"%x
def f2(x): return "%.2f"%x
out=[]
def emit(s): out.append(s)

# ---- S1 decision distribution ----
emit(r"\newcommand{\tableSdecdist}{%")
for m in ORDER:
    d=D["decdist"][m]
    emit(f"{NAME[m]} & {f1(100-d['refNO'])} & {f1(d['refNO'])} & {f1(100-d['qNO'])} & {f1(d['qNO'])} & {f1(100-d['ref_ner'])}/{f1(100-d['ref_qar'])} & {f1(100-d['q_ner'])}/{f1(100-d['q_qar'])} \\\\")
emit("}")

# ---- S2 per-config marginal (certCov [95% CI] ; agreement) ----
emit(r"\newcommand{\tableSperconfig}{%")
cfgs=["RTN-8","RTN-6","RTN-4","INT8","NF4"]
for m in ORDER:
    pc=D["perconfig"][m]; ci=A["ci_perconfig"][m]
    cells=" & ".join(f"{f1(pc[c]['certCov'])}\\,[{f1(ci[c][0])},{f1(ci[c][1])}]" for c in cfgs)
    emit(f"{NAME[m]} & {cells} \\\\")
emit("}")

# ---- S3 NF4 marginal decision-risk audit (coverage with patient-clustered 95% CI) ----
emit(r"\newcommand{\tableSaudit}{%")
for m in ORDER:
    a=D["audit"][m]; ci=A["ci_audit"][m]["covCI"]
    emit(f"{NAME[m]} & {a['ncal']} & {a['xcal']} & {f2(a['tau'])} & {f1(a['cp'])} & {f1(a['certCov'])} [{f1(ci[0])}, {f1(ci[1])}] \\\\")
emit("}")

# ---- S4 Mondrian per-class decision-risk audit (n, x, tau, cov, CP per class) ----
emit(r"\newcommand{\tableSmond}{%")
for m in ORDER:
    p=D["mond"][m]
    tY="$\\infty$" if p['tauY']==float('inf') else f2(p['tauY'])
    tN="$\\infty$" if p['tauN']==float('inf') else f2(p['tauN'])
    covN="\\emph{esc}" if p['covN']==0.0 else f1(p['covN']); cpN="--" if p['covN']==0.0 else f1(p['cpN'])
    emit(f"{NAME[m]} & {p['nY']} & {p['xY']} & {tY} & {f1(p['covY'])} & {f1(p['cpY'])} & {p['nN']} & {p['xN']} & {tN} & {covN} & {cpN} \\\\")
emit("}")

# ---- S(new) class-specific calibration resolution + escalation reason ----
emit(r"\newcommand{\tableSclassres}{%")
RLAB={"certified":"certified","resolution-limited":"\\emph{resolution-limited}","risk-limited":"\\emph{risk-limited}"}
for m in ORDER:
    c=A["classres"][m]
    emit(f"{NAME[m]} & {c['nY']} & {c['nN']} & {f2(c['floorY'])} & {f2(c['floorN'])} & {f1(A3['mincpN'][m])} & {RLAB[A3['reason'][m]]} \\\\")
emit("}")

# ---- threshold-grid sensitivity ----
emit(r"\newcommand{\tableSgrid}{%")
for G in [25,50,100,200]:
    g=A3["grid"][G]
    emit(f"{G} & {f1(g['lo'])} & {f1(g['med'])} & {f1(g['hi'])} & {f1(g['flag']['qwen2.5-72b'])} & {f1(g['flag']['huatuogpt-o1-72b'])} & {f1(g['flag']['llama3.3-70b'])} \\\\")
emit("}")

# ---- repeated patient-disjoint split stability ----
emit(r"\newcommand{\tableSsplit}{%")
for m in ORDER:
    s=A3["split"][m]
    emit(f"{NAME[m]} & {f1(s['med'])} & {f1(s['iqr'][0])}--{f1(s['iqr'][1])} & {f1(s['lo'])} & {f1(s['hi'])} \\\\")
emit("}")

# ---- merged confidence-budget sensitivity (marginal + Mondrian) ----
emit(r"\newcommand{\tableSconfidence}{%")
for m in ORDER:
    mg=D["delta"][m]; mo=A3["monddelta"][m]
    emit(f"{NAME[m]} & {f1(mg[0.10])} & {f1(mg[0.05])} & {f1(mg[0.01])} & {f1(mo[0.10])} & {f1(mo[0.05])} & {f1(mo[0.01])} \\\\")
emit("}")

# ---- class-conditional alpha=1/5/10 frontier (size + escalation) ----
emit(r"\newcommand{\tableSmondalpha}{%")
for m in ORDER:
    a=A3["mondalpha"][m]
    emit(f"{NAME[m]} & {f1(a[0.01]['size'])} & {f1(a[0.01]['esc'])} & {f1(a[0.05]['size'])} & {f1(a[0.05]['esc'])} & {f1(a[0.10]['size'])} & {f1(a[0.10]['esc'])} \\\\")
emit("}")

# ---- split-sample exactness: full-calibration vs split-sample frontier ----
emit(r"\newcommand{\tableSexact}{%")
for m in ORDER:
    full=D["mond"][m]; split=D["mond_split"][m]
    cfg="NF4" if (full["q"]=="bnb" and full["bit"]==4) else "%s-%d"%(full["q"].upper(),full["bit"])
    emit(f"{NAME[m]} & {f1(full['size'])} & {f1(split['size'])} & {'%+.1f'%(split['size']-full['size'])} & {cfg} & Yes \\\\")
emit("}")

# ---- S6 calibration size ----
emit(r"\newcommand{\tableScalsize}{%")
for nc in sorted(D["calsize"]):
    c=D["calsize"][nc]
    emit(f"{nc} & {f1(c['lo'])} & {f1(c['med'])} & {f1(c['hi'])} \\\\")
emit("}")

# ---- S7 task-wise (+ range across models) ----
emit(r"\newcommand{\tableStaskwise}{%")
for c in ["RTN-8","RTN-6","RTN-4","INT8","NF4"]:
    t=D["taskwise"][c]
    emit(f"{c} & {f1(t['nerCov'])} & {f1(t['nerDis'])} & {f1(t['qaCov'])} & {f1(t['qaDis'])} \\\\")
emit("}")

# ---- S8 external NER ----
emit(r"\newcommand{\tableSner}{%")
def g(m,ds,q,c):
    return D["ner"].get((m,ds),{}).get((q,c))
for ds,dsname in [("bc5cdr_disease","BC5CDR-Disease"),("ncbi_disease","NCBI-Disease")]:
    emit(f"\\multicolumn{{7}}{{l}}{{\\textit{{{dsname}}}}}\\\\")
    for m in ORDER:
        full=g(m,ds,"full","full"); r8=g(m,ds,"rtn","8"); r6=g(m,ds,"rtn","6"); r4=g(m,ds,"rtn","4"); nf=g(m,ds,"bnb","4")
        if full is None: continue
        emit(f"{NAME[m]} & {'%.3f'%full[2]} & {'%.3f'%r8[2]} & {'%.3f'%r6[2]} & {'%.3f'%r4[2]} & {'%.3f'%nf[2]} & {'%.3f'%nf[0]}/{'%.3f'%nf[1]} \\\\")
emit("}")

# ---- S9 prevalence-shift, all 11 models + max/mean ----
emit(r"\newcommand{\tableSprev}{%")
QS=D["prev_qs"]
emit(r"\textbf{Maximum over models} & marginal & "+" & ".join(f1(A['prev_max']['marg'][q]) for q in QS)+r" \\")
emit(r" & Mondrian & "+" & ".join(f1(A['prev_max']['mond'][q]) for q in QS)+r" \\")
emit(r"\textbf{Mean over models} & marginal & "+" & ".join(f1(D['prev_agg']['marg'][q]) for q in QS)+r" \\")
emit(r" & Mondrian & "+" & ".join(f1(D['prev_agg']['mond'][q]) for q in QS)+r" \\")
emit(r"\midrule")
for m in ORDER:
    pm=D["prev_model"][m]
    emit(f"{NAME[m]} & marginal & "+" & ".join(f1(x) for x in pm['marg'])+r" \\")
    emit(f" & Mondrian & "+" & ".join(f1(x) for x in pm['mond'])+r" \\")
emit("}")

open("si_tables.tex","w").write("\n".join(out)+"\n")
print("wrote si_tables.tex ("+str(len(out))+" lines)")
