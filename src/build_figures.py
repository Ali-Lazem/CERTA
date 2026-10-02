#!/usr/bin/env python3
"""
build_figures.py -- builds the empirical main-text figures for the CERTA manuscript as vector PDFs
(fig2.pdf ... fig5.pdf). Figure 1 is the CERTA architecture, built separately from the standalone
TikZ file fig1_architecture.tex (-> fig1.pdf). Self-contained: every number is embedded and matches
the verified analysis (cert_all_v2.csv, the patient-disjoint / Mondrian frontier, bench.csv).
Requires only matplotlib and numpy. Design uses dumbbells, slopegraphs, a heatmap and a
memory-zone chart (no bar charts).

    python build_figures.py         # writes fig2.pdf ... fig5.pdf into the working directory
"""
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "legend.frameon": False, "axes.axisbelow": True,
})
FAM = {"Llama": "#1f4e79", "Qwen": "#2e7d32", "Mistral": "#b3202c"}
INK, GRID = "#222222", "#dddddd"
def fam_of(m):
    m = m.lower()
    return "Qwen" if ("qwen" in m or "huatuo" in m) else ("Mistral" if "mistral" in m else "Llama")
def panel(ax, letter):
    # offset in POINTS above the axes top-left, so it clears the title on both tall and short panels
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction",
                xytext=(-28, 22), textcoords="offset points",
                fontweight="bold", fontsize=12, va="bottom", ha="left", annotation_clip=False)
SHORT = {"llama3.3-70b":"Llama-3.3-70B","llama3-70b-base":"Llama-3-70B","mmed-3.1-70b":"MMed-3.1-70B",
"openbiollm-70b":"OpenBioLLM-70B","med42-70b":"Med42-70B","qwen2.5-72b":"Qwen2.5-72B",
"huatuogpt-o1-72b":"HuatuoGPT-72B","llama3-8b":"Llama-3-8B","mmed-8b":"MMed-8B",
"mistral-7b":"Mistral-7B","biomistral-7b":"BioMistral-7B"}
ORDER = ["qwen2.5-72b","huatuogpt-o1-72b","med42-70b","openbiollm-70b","mmed-3.1-70b","llama3.3-70b",
         "llama3-70b-base","llama3-8b","mmed-8b","mistral-7b","biomistral-7b"]

# ---------------- data computed from certa_core (split-primary single source of truth) ----------------
import certa_core as C
import csv as _csv
ALPHAS = list(range(1,11))
_CFG={"rtn8":("rtn",8),"rtn6":("rtn",6),"rtn4":("rtn",4),"int8":("bnb",8),"nf4":("bnb",4)}
CERT={}
for m in ORDER:
    CERT[m]={}
    for k,(q,b) in _CFG.items():
        r=C.marg(m,q,b); CERT[m][k]=(round(r["rawPres"],1),round(r["certCov"],1))
def mon_env(m): return [C.mond_frontier(m,alpha=a/100.0)["size"] for a in ALPHAS]   # deployable Mondrian size vs alpha
MARG={m:round(C.marg_frontier(m)[2]["size"],1) for m in ORDER}                       # marginal deployable frontier at alpha=5
CALIB={}
for m in ORDER:
    for q,b,tag in [("bnb",4,"bnb"),("bnb",8,"int8"),("rtn",4,"rtn")]:
        ref,d,mg=C.arr(m,q,b); fl=(d!=ref).astype(int); ser=[]
        for a in ALPHAS:
            tau=C._tau(mg,fl,C.SEL,C.CERT,a/100.0,C.DELTA); cov=mg[C.TEST]>=tau
            ser.append(round(100*fl[C.TEST][cov].mean(),2) if cov.sum() else 0.0)
        CALIB[f"{m}_{tag}"]=ser
VRAM = {"llama3.3-70b":(141.9,73.5,41.2),"mmed-3.1-70b":(141.9,73.5,41.2),"llama3-70b-base":(141.9,73.5,41.2),
"openbiollm-70b":(141.7,73.3,41.0),"med42-70b":(141.9,73.5,41.2),"qwen2.5-72b":(146.3,77.6,43.0),
"huatuogpt-o1-72b":(146.1,77.5,42.7),"llama3-8b":(16.7,9.7,6.4),"mmed-8b":(16.7,12.4,8.7),
"mistral-7b":(14.9,8.5,4.7),"biomistral-7b":(14.8,8.4,4.6)}
# external disease-NER F1 from bench.csv: NER[m][step] = (BC5CDR F1, NCBI F1)
NER={m:{v:[None,None] for v in ("full","r8","r6","r4","nf4")} for m in ORDER}
_DS={"bc5cdr_disease":0,"ncbi_disease":1}; _CK={("full","full"):"full",("rtn","8"):"r8",("rtn","6"):"r6",("rtn","4"):"r4",("bnb","4"):"nf4"}
for _r in _csv.DictReader(open("verify/bench.csv")):
    _m=_r["model"]; _d=_DS.get(_r["dataset"]); _c=_CK.get((_r["quantizer"],_r["config"]))
    if _m in NER and _d is not None and _c: NER[_m][_c][_d]=float(_r["f1"])
NER={m:{k:tuple(v) for k,v in d.items()} for m,d in NER.items()}
# annotation numbers, computed
_rawcert=[CERT[m][c][0]-CERT[m][c][1] for m in ORDER for c in ("rtn4","nf4")]
GAP_MEAN=round(sum(_rawcert)/len(_rawcert),1)                                        # mean raw-minus-certified gap
_mm=[mon_env(m)[4]-MARG[m] for m in ORDER]; MON_LO,MON_HI=round(min(_mm),1),round(max(_mm),1)
FLAG_MOND=round(mon_env("qwen2.5-72b")[4],1)                                         # flagship Qwen Mondrian at alpha=5

# ========================================================================================
# FIGURE 2 -- certification landscape (heatmap a) + certificate-vs-raw (b) + hardware (c)
#   The CERTA architecture (old Fig 1a) is now a standalone TikZ figure (fig1.pdf); its
#   VRAM panel moves here as panel c. Produces fig2.pdf.
# ========================================================================================
def figure2():
    fig = plt.figure(figsize=(7.4, 6.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.16, 1.0], height_ratios=[1.0, 0.9],
                          hspace=0.52, wspace=0.52)
    # --- panel a: certified-coverage heatmap (spans both rows, left) ---
    ax = fig.add_subplot(gs[:, 0]); cfgs=["rtn8","rtn6","rtn4","int8","nf4"]; labels=["RTN-8","RTN-6","RTN-4","INT8","NF4-4"]
    M=np.array([[CERT[m][c][1] for c in cfgs] for m in ORDER])
    im=ax.imshow(M,aspect="auto",cmap="YlGnBu",vmin=0,vmax=100)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels,rotation=30,ha="right",fontsize=7.5)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([SHORT[m] for m in ORDER],fontsize=7.5)
    for i,m in enumerate(ORDER):
        for j,c in enumerate(cfgs):
            v=CERT[m][c][1]; ax.text(j,i,f"{v:.0f}",ha="center",va="center",fontsize=6.6,color="white" if v>60 else "#333")
    for i,m in enumerate(ORDER):
        if CERT[m]["int8"][1]==0.0: ax.text(3,i,"$\\times$",ha="center",va="center",fontsize=11,color="#b3202c")
        if CERT[m]["nf4"][1]==100.0: ax.plot(4.34,i-0.32,marker="*",ms=7,color="#ffd400",mec="#333",mew=0.4,clip_on=False)
    cb=fig.colorbar(im,ax=ax,orientation="horizontal",fraction=0.043,pad=0.09)
    cb.set_label("certified coverage (%)",fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title("Marginal certified coverage across configurations",fontsize=8.9,pad=6); panel(ax,"a")
    # --- panel b: raw vs certified (top-right) ---
    axb=fig.add_subplot(gs[0,1])
    for m in ORDER:
        for c,mk in [("rtn4","o"),("nf4","s")]:
            raw,cov=CERT[m][c]; axb.scatter(raw,cov,s=27,color=FAM[fam_of(m)],marker=mk,edgecolor="white",linewidth=0.5,zorder=3,alpha=.92)
    axb.plot([40,100],[40,100],ls="--",lw=0.9,color="#888"); axb.text(98,99,"$y=x$",fontsize=7,color="#888",ha="right")
    axb.annotate("Llama-3-8B\nRTN-4",(CERT["llama3-8b"]["rtn4"][0],CERT["llama3-8b"]["rtn4"][1]),(54,32),fontsize=6.3,color=INK,arrowprops=dict(arrowstyle="->",lw=0.7,color="#666"))
    axb.set_xlabel("raw decision preservation (%)",fontsize=8); axb.set_ylabel("certified coverage (%)",fontsize=8)
    axb.set_xlim(40,101); axb.set_ylim(0,106); axb.grid(color=GRID,lw=0.6)
    axb.set_title("Raw agreement is not a deployment certificate",fontsize=8.6)
    leg=[Line2D([0],[0],marker="o",color="w",markerfacecolor="#666",label="RTN 4-bit",ms=6),
         Line2D([0],[0],marker="s",color="w",markerfacecolor="#666",label="NF4 4-bit",ms=6)]+\
        [Line2D([0],[0],marker="o",color="w",markerfacecolor=c,label=f,ms=6) for f,c in FAM.items()]
    axb.legend(handles=leg,loc="lower right",fontsize=6.1,labelspacing=0.25,borderpad=0.3); panel(axb,"b")
    # --- panel c: memory-zone transition for the 70-72B models (bottom-right) ---
    axc=fig.add_subplot(gs[1,1])
    big=sorted(["qwen2.5-72b","huatuogpt-o1-72b","med42-70b","openbiollm-70b","mmed-3.1-70b","llama3.3-70b","llama3-70b-base"],
               key=lambda m: VRAM[m][0])
    axc.axvspan(0,48,color="#e8f4ea",zorder=0); axc.axvspan(48,80,color="#fdf3e3",zorder=0); axc.axvspan(80,158,color="#fbeceb",zorder=0)
    for x,txt,c in [(24,"1$\\times$48 GB","#2e7d32"),(64,"1$\\times$80","#8a6d3b"),(119,"2 GPUs","#b3202c")]:
        axc.text(x,len(big)+0.12,txt,ha="center",va="bottom",fontsize=6.0,color=c,fontweight="bold")
    for i,m in enumerate(big):
        bf,i8,n4=VRAM[m]
        axc.plot([n4,bf],[i,i],"-",color="#cccccc",lw=2.0,zorder=2,solid_capstyle="round")
        axc.scatter(n4,i,s=34,color="#2e7d32",edgecolor="white",lw=0.5,zorder=5)
        axc.scatter(i8,i,s=20,color="#8a6d3b",edgecolor="white",lw=0.4,zorder=4)
        axc.scatter(bf,i,s=34,color="#1f4e79",edgecolor="white",lw=0.5,zorder=4)
    axc.axvline(48,ls="--",lw=0.8,color="#2e7d32"); axc.axvline(80,ls="--",lw=0.8,color="#8a6d3b")
    axc.set_yticks(range(len(big))); axc.set_yticklabels([SHORT[m] for m in big],fontsize=6.6)
    axc.set_xlim(0,158); axc.set_ylim(-0.6,len(big)+0.95); axc.set_xlabel("peak VRAM (GB)",fontsize=8)
    axc.grid(axis="x",color=GRID,lw=0.5)
    leg2=[Line2D([0],[0],marker="o",color="w",markerfacecolor="#1f4e79",label="BF16",ms=6),
          Line2D([0],[0],marker="o",color="w",markerfacecolor="#8a6d3b",label="INT8",ms=5),
          Line2D([0],[0],marker="o",color="w",markerfacecolor="#2e7d32",label="NF4",ms=6)]
    axc.legend(handles=leg2,loc="lower left",fontsize=5.8,ncol=1,handletextpad=0.3,labelspacing=0.25,borderpad=0.3)
    axc.set_title("Hardware footprint of the 70--72B models",fontsize=8.6,pad=6); panel(axc,"c")
    fig.savefig("fig2.pdf"); plt.close(fig)

# ========================================================================================
# FIGURE 3 -- risk-resource lines (a) + calibration (b) + marginal-vs-Mondrian DUMBBELL (c)
# ========================================================================================
def figure3():
    fig = plt.figure(figsize=(7.3,5.4)); gs = fig.add_gridspec(2,2,height_ratios=[1,1.02],hspace=0.42,wspace=0.30)
    axa=fig.add_subplot(gs[0,0]); rep=["qwen2.5-72b","huatuogpt-o1-72b","med42-70b","mmed-3.1-70b","llama3.3-70b","biomistral-7b"]
    for m in rep: axa.plot(ALPHAS,mon_env(m),"o-",ms=3,lw=1.5,color=FAM[fam_of(m)],alpha=.9,label=SHORT[m])
    axa.axhline(25,ls=":",lw=0.9,color="#666"); axa.text(9.7,26,"4$\\times$ (25%)",fontsize=6.6,color="#666",ha="right")
    axa.set_xlabel("risk budget $\\alpha$ (%)"); axa.set_ylabel("deployment size (% BF16)")
    axa.set_title("Class-conditional risk--resource frontier",fontsize=9.0); axa.grid(color=GRID,lw=0.6)
    axa.legend(fontsize=6.0,ncol=2,loc="upper right"); panel(axa,"a")
    axb=fig.add_subplot(gs[0,1])
    for k,v in CALIB.items(): axb.plot(ALPHAS,v,"-",lw=0.8,color="#8fb0c9",alpha=.75,zorder=2)
    axb.plot([1,10],[1,10],"k--",lw=1.0,zorder=4,label="target = observed")
    axb.fill_between([1,10],[1,10],[10,10],color="#b3202c",alpha=0.06,zorder=1)
    axb.text(6.3,8.7,"observed > target",fontsize=6.6,color="#b3202c")
    axb.set_xlabel("target risk $\\alpha$ (%)"); axb.set_ylabel("observed disagreement (%)")
    axb.set_xlim(1,10); axb.set_ylim(0,11); axb.set_title("Calibration diagnostic",fontsize=9.0)
    axb.grid(color=GRID,lw=0.6); axb.legend(loc="upper left",fontsize=6.6); panel(axb,"b")
    # c: dumbbell marginal -> Mondrian at alpha=5, sorted by Mondrian
    axc=fig.add_subplot(gs[1,:]); mon={m:mon_env(m)[4] for m in ORDER}
    order2=sorted(ORDER,key=lambda m:mon[m]); y=np.arange(len(order2))
    esc={"biomistral-7b","openbiollm-70b","mmed-3.1-70b"}
    for i,m in enumerate(order2):
        axc.plot([MARG[m],mon[m]],[i,i],"-",color="#c9c9c9",lw=2.4,zorder=2,solid_capstyle="round")
        axc.scatter(MARG[m],i,s=42,facecolor="white",edgecolor="#3182bd",lw=1.6,zorder=3)
        axc.scatter(mon[m],i,s=46,color="#3182bd",edgecolor="white",lw=0.6,zorder=4)
        if m in esc: axc.text(mon[m]+0.7,i,"reject class escalated",fontsize=5.9,va="center",color="#b3202c")
    axc.axvline(25,ls=":",lw=0.9,color="#666"); axc.text(25,len(order2)-0.2,"4$\\times$",fontsize=6.6,color="#666",ha="center")
    axc.set_yticks(y); axc.set_yticklabels([SHORT[m] for m in order2],fontsize=6.8)
    axc.set_xlabel("deployment size at $\\alpha=5\\%$ (% BF16)"); axc.set_xlim(22,48); axc.set_ylim(-0.6,len(order2)+0.1)
    axc.grid(axis="x",color=GRID,lw=0.6)
    axc.set_title("Class-conditional (Mondrian) vs marginal: $%+.1f$ to $%+.1f$ pp; flagship near %.0f%%"%(MON_LO,MON_HI,FLAG_MOND),fontsize=8.6)
    leg=[Line2D([0],[0],marker="o",color="w",markerfacecolor="white",markeredgecolor="#3182bd",label="marginal",ms=7,mew=1.5),
         Line2D([0],[0],marker="o",color="w",markerfacecolor="#3182bd",label="class-conditional (Mondrian)",ms=7)]
    axc.legend(handles=leg,loc="lower right",fontsize=6.8); panel(axc,"c")
    fig.savefig("fig3.pdf"); plt.close(fig)

# ========================================================================================
# FIGURE 4 -- RTN->NF4 dumbbells: raw preservation (a) and certified coverage (b)
# ========================================================================================
def figure4():
    fig,(axa,axb)=plt.subplots(1,2,figsize=(7.3,4.3),gridspec_kw=dict(wspace=0.46))
    shared=sorted(ORDER,key=lambda m:CERT[m]["nf4"][1])  # both panels share one row order
    def dumbbell(ax,getr,getn,title,xlabel,vline=None):
        order2=shared; y=np.arange(len(order2))
        for i,m in enumerate(order2):
            r,n=getr(m),getn(m)
            ax.plot([r,n],[i,i],"-",color="#cfcfcf",lw=2.2,zorder=2,solid_capstyle="round")
            ax.annotate("",xy=(n,i),xytext=(r,i),arrowprops=dict(arrowstyle="-|>",lw=0,color="#cfcfcf"))
            ax.scatter(r,i,s=40,facecolor="white",edgecolor="#888",lw=1.4,zorder=3)
            ax.scatter(n,i,s=44,color=FAM[fam_of(m)],edgecolor="white",lw=0.6,zorder=4)
        if vline is not None: ax.axvline(vline,ls=":",lw=0.9,color="#666")
        ax.set_yticks(y); ax.set_yticklabels([SHORT[m] for m in order2],fontsize=7)
        ax.set_xlabel(xlabel); ax.set_title(title,fontsize=9.0); ax.grid(axis="x",color=GRID,lw=0.6)
        ax.set_ylim(-0.6,len(order2)+0.1)
    dumbbell(axa,lambda m:CERT[m]["rtn4"][0],lambda m:CERT[m]["nf4"][0],
             "Raw 4-bit preservation: RTN $\\to$ NF4","decision preservation (%)")
    axa.set_xlim(45,101); panel(axa,"a")
    dumbbell(axb,lambda m:CERT[m]["rtn4"][1],lambda m:CERT[m]["nf4"][1],
             "Certified 4-bit coverage: RTN $\\to$ NF4","certified coverage (%)",vline=95)
    axb.set_xlim(-3,105); panel(axb,"b")
    leg=[Line2D([0],[0],marker="o",color="w",markerfacecolor="white",markeredgecolor="#888",label="RTN 4-bit",ms=7,mew=1.4),
         Line2D([0],[0],marker="o",color="w",markerfacecolor="#555",label="NF4 4-bit",ms=7)]
    axa.legend(handles=leg,loc="lower left",fontsize=6.8)
    fig.savefig("fig4.pdf"); plt.close(fig)

# ========================================================================================
# FIGURE 5 -- external NER slopegraphs (a,b) + coverage-vs-|dF1| scatter (c)
# ========================================================================================
def figure5():
    fig=plt.figure(figsize=(7.3,5.3)); gs=fig.add_gridspec(2,2,height_ratios=[1,1],hspace=0.5,wspace=0.28)
    steps=["full","r8","r6","r4","nf4"]; xlab=["full","RTN-8","RTN-6","RTN-4","NF4-4"]; xs=range(5)
    def slope(ax,ds,title):
        for m in ORDER:
            ys=[NER[m][s][ds] for s in steps]; c=FAM[fam_of(m)]
            hl = m in ("mmed-3.1-70b",)
            ax.plot(xs,ys,"-",color=c,lw=2.0 if hl else 1.0,alpha=1.0 if hl else 0.5,zorder=4 if hl else 3)
            ax.scatter(xs,ys,s=14 if hl else 8,color=c,zorder=5,alpha=1.0 if hl else 0.6)
        ax.annotate("MMed-3.1: RTN-4 drop,\nNF4 recovery",(3,NER["mmed-3.1-70b"]["r4"][ds]),
                    (1.2,0.10),fontsize=6.0,color=FAM["Llama"],arrowprops=dict(arrowstyle="->",lw=0.7,color=FAM["Llama"]))
        ax.set_xticks(list(xs)); ax.set_xticklabels(xlab,rotation=22,ha="right",fontsize=7)
        ax.set_ylabel("F1"); ax.set_ylim(0,1.0); ax.set_xlim(-0.2,4.2); ax.grid(color=GRID,lw=0.6); ax.set_title(title,fontsize=9.0)
    axa=fig.add_subplot(gs[0,0]); slope(axa,0,"BC5CDR-Disease"); panel(axa,"a")
    axb=fig.add_subplot(gs[0,1]); slope(axb,1,"NCBI-Disease"); panel(axb,"b")
    leg=[Line2D([0],[0],color=c,lw=2,label=f) for f,c in FAM.items()]
    axb.legend(handles=leg,loc="upper right",fontsize=6.6)
    axc=fig.add_subplot(gs[1,:])
    lab={"openbiollm-70b":(88.0,0.084,"right"),"med42-70b":(88.0,0.074,"right"),
         "mmed-3.1-70b":(88.0,0.064,"right"),"huatuogpt-o1-72b":(96.2,0.047,"right"),
         "qwen2.5-72b":(98.8,0.006,"right")}
    for m in ORDER:
        cov=CERT[m]["nf4"][1]; d=np.mean([abs(NER[m]["nf4"][k]-NER[m]["full"][k]) for k in (0,1)])
        axc.scatter(cov,d,s=46,color=FAM[fam_of(m)],edgecolor="white",lw=0.5,zorder=3)
        if m in lab:
            tx,ty,ha=lab[m]
            axc.annotate(SHORT[m],(cov,d),(tx,ty),fontsize=6.2,ha=ha,va="center",color=INK,
                         arrowprops=dict(arrowstyle="-",lw=0.6,color="#aaaaaa"))
    _cx=[CERT[m]["nf4"][1] for m in ORDER]; _cy=[np.mean([abs(NER[m]["nf4"][k]-NER[m]["full"][k]) for k in (0,1)]) for m in ORDER]
    _rr=float(np.corrcoef(_cx,_cy)[0,1])
    axc.set_xlabel("NF4 4-bit certified coverage (%)"); axc.set_ylabel("|$\\Delta$F1| vs full (NF4 4-bit)")
    axc.set_title("Reference agreement and task accuracy are complementary ($r\\approx%.1f$)"%_rr,fontsize=9.0,pad=9)
    axc.grid(color=GRID,lw=0.6); axc.set_xlim(78,103); axc.set_ylim(-0.004,0.095)
    axc.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor=c,label=f,ms=6) for f,c in FAM.items()],
               loc="upper right",fontsize=6.8); panel(axc,"c")
    fig.savefig("fig5.pdf"); plt.close(fig)

if __name__ == "__main__":
    # fig1.pdf is the TikZ architecture (fig1_architecture.tex), built separately in LaTeX.
    figure2(); figure3(); figure4(); figure5()
    print("wrote fig2.pdf fig3.pdf fig4.pdf fig5.pdf (fig1.pdf = TikZ architecture)")
