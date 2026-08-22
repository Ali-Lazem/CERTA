#!/usr/bin/env python3
"""Supplementary figures for CERTA. Filenames match the compiled figure numbers:
  figS1 = full marginal frontiers   figS2 = prevalence shift
  figS3 = escalated-case analysis   figS4 = class-specific resolution
  figS5 = statistical robustness (delta + cal size)   figS6 = split/grid stability
Loads si_data.pkl, si_add.pkl, si_add3.pkl and the frontier CSVs."""
import pickle, csv, numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"figure.dpi":200,"savefig.dpi":300,"savefig.bbox":"tight","font.family":"sans-serif",
"font.sans-serif":["DejaVu Sans"],"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
"axes.linewidth":0.8,"legend.frameon":False})
FAM={"Llama":"#1f4e79","Qwen":"#2e7d32","Mistral":"#b3202c"}; GRID="#dddddd"; INK="#222"
RC={"certified":"#2e7d32","risk-limited":"#d97706","resolution-limited":"#b3202c"}
SHORT={"llama3.3-70b":"Llama-3.3-70B","llama3-70b-base":"Llama-3-70B","mmed-3.1-70b":"MMed-3.1-70B",
"openbiollm-70b":"OpenBioLLM-70B","med42-70b":"Med42-70B","qwen2.5-72b":"Qwen2.5-72B",
"huatuogpt-o1-72b":"HuatuoGPT-72B","llama3-8b":"Llama-3-8B","mmed-8b":"MMed-8B",
"mistral-7b":"Mistral-7B","biomistral-7b":"BioMistral-7B"}
ORDER=["qwen2.5-72b","huatuogpt-o1-72b","med42-70b","openbiollm-70b","mmed-3.1-70b","llama3.3-70b",
"llama3-70b-base","llama3-8b","mmed-8b","mistral-7b","biomistral-7b"]
def fam(m): m=m.lower(); return "Qwen" if("qwen"in m or "huatuo"in m) else "Mistral" if "mistral"in m else "Llama"
def panel(ax,l): ax.text(-0.13,1.06,l,transform=ax.transAxes,fontweight="bold",fontsize=12,va="top")
D=pickle.load(open("si_data.pkl","rb")); AD=pickle.load(open("si_add.pkl","rb")); AD3=pickle.load(open("si_add3.pkl","rb"))
ALPHAS=list(range(1,11))

# marginal frontier envelope per (model, alpha) from the frontier CSVs
def load_frontier():
    env={}
    for f in ("verify/frontier_rtn.csv","verify/frontier_bnb.csv"):
        for r in csv.DictReader(open(f)):
            m=r["model"].rsplit("_",1)[0]; a=int(round(float(r["alpha"])*100))
            env.setdefault((m,a),1e9); env[(m,a)]=min(env[(m,a)],float(r["sizePct"]))
    return env
ENV=load_frontier()

# ========================= Fig S1: full marginal frontiers =========================
def fig_frontiers():
    fig,ax=plt.subplots(figsize=(7.0,4.2))
    for m in ORDER:
        ys=[ENV[(m,a)] for a in ALPHAS]
        ax.plot(ALPHAS,ys,"o-",ms=3.2,lw=1.4,color=FAM[fam(m)],alpha=.85,label=SHORT[m])
    ax.axhline(25,ls=":",lw=0.9,color="#666"); ax.text(9.7,25.8,"4$\\times$ (25%)",fontsize=7,color="#666",ha="right")
    ax.set_xlabel("risk budget $\\alpha$ (%)"); ax.set_ylabel("marginal deployment size (% BF16)")
    ax.set_title("Certified compression frontiers, all eleven models (marginal certificate)",fontsize=9.5)
    ax.grid(color=GRID,lw=0.6); ax.set_xticks(ALPHAS); ax.legend(fontsize=6.4,ncol=2,loc="upper right")
    fig.savefig("figS1.pdf"); plt.close(fig)

# ========================= Fig S2: prevalence-shift =========================
def fig_prevalence():
    x=[5,10,20,30,40,50]                                  # deployment NO prevalence (%), increasing
    hi={"openbiollm-70b":"#1f4e79","llama3-70b-base":"#d97706","qwen2.5-72b":"#2e7d32"}
    fig,(axa,axb)=plt.subplots(1,2,figsize=(7.4,3.9),sharey=True,gridspec_kw=dict(wspace=0.07))
    for ax,key,title in [(axa,"marg","Marginal certificate"),(axb,"mond","Class-conditional (Mondrian)")]:
        ax.axhspan(5,11,color="#b3202c",alpha=0.075,zorder=0)
        ax.axhline(5,ls="--",lw=1.1,color="#b3202c",zorder=2)
        for m in ORDER:                                   # all eleven models, light grey
            if m in hi: continue
            ax.plot(x,D["prev_model"][m][key][::-1],"-",color="#bcbcbc",lw=1.0,alpha=.9,zorder=1)
        for m,c in hi.items():                            # highlighted trajectories
            ax.plot(x,D["prev_model"][m][key][::-1],"o-",color=c,lw=2.1,ms=4.2,zorder=4,
                    markeredgecolor="white",markeredgewidth=0.5)
        ax.set_xlabel("deployment NO (reject) prevalence (%)"); ax.set_xlim(3,52); ax.set_ylim(0,10.7)
        ax.set_xticks(x); ax.grid(color=GRID,lw=0.6); ax.set_title(title,fontsize=9.6)
    axa.set_ylabel("certified-set disagreement (%)")
    axa.text(51,10.2,"risk-budget violation",fontsize=6.8,color="#b3202c",ha="right",va="top",style="italic")
    axb.text(51,5.35,"$\\alpha=5\\%$ target",fontsize=6.8,color="#b3202c",ha="right",va="bottom")
    panel(axa,"a"); panel(axb,"b")
    leg=[Line2D([0],[0],color=c,lw=2.4,label=SHORT[m]) for m,c in hi.items()]+\
        [Line2D([0],[0],color="#bcbcbc",lw=1.4,label="other eight models")]
    axb.legend(handles=leg,fontsize=6.8,loc="upper right",borderpad=0.4,labelspacing=0.3)
    fig.savefig("figS2.pdf",bbox_inches="tight"); plt.close(fig)

# ========================= Fig S3: escalated-case analysis =========================
def fig_escalated():
    fig,(axa,axb)=plt.subplots(1,2,figsize=(7.3,4.0),gridspec_kw=dict(wspace=0.34))
    # panel a: ECDF of decision margins, escalated vs certified (pooled over models, NF4)
    for a_,c,lab in [(AD["ecdf_esc"],"#b3202c","escalated"),(AD["ecdf_cert"],"#1f4e79","certified")]:
        xs=np.sort(a_); ys=np.arange(1,len(xs)+1)/len(xs); axa.plot(xs,ys,"-",color=c,lw=2.1,label=lab)
    axa.set_xlim(0,18); axa.set_ylim(0,1.02); axa.set_xlabel("decision margin $m_k(x)$")
    axa.set_ylabel("cumulative fraction of decisions"); axa.set_title("Margin distribution (NF4, pooled)",fontsize=9.2)
    axa.grid(color=GRID,lw=0.6); axa.legend(fontsize=7.6,loc="lower right")
    axa.text(0.6,0.62,"escalated\nmedian $0.35$",fontsize=6.6,color="#b3202c")
    axa.text(6.3,0.45,"certified\nmedian $5.12$",fontsize=6.6,color="#1f4e79")
    # panel b: disagreement dumbbell, certified vs escalated
    order=sorted([m for m in ORDER if not np.isnan(D["escalated"][m]["disEsc"])],key=lambda m:D["escalated"][m]["disEsc"])
    y=np.arange(len(order))
    for i,m in enumerate(order):
        e=D["escalated"][m]
        axb.plot([e["disCert"],e["disEsc"]],[i,i],"-",color="#cfcfcf",lw=2.2,zorder=2,solid_capstyle="round")
        axb.scatter(e["disCert"],i,s=44,color=FAM[fam(m)],edgecolor="white",lw=0.5,zorder=4)
        axb.scatter(e["disEsc"],i,s=40,facecolor="white",edgecolor="#b3202c",lw=1.3,zorder=3)
    axb.axvline(5,ls=":",lw=0.9,color="#666"); axb.set_yticks(y); axb.set_yticklabels([SHORT[m] for m in order],fontsize=7)
    axb.set_xlabel("disagreement with BF16 reference (%)"); axb.set_title("Disagreement: certified vs escalated",fontsize=9)
    axb.grid(axis="x",color=GRID,lw=0.6); axb.set_ylim(-.6,len(order))
    axb.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor="#555",label="certified",ms=7),
                        Line2D([0],[0],marker="o",color="w",markerfacecolor="white",markeredgecolor="#b3202c",label="escalated",ms=7,mew=1.3)],loc="lower right",fontsize=6.8)
    panel(axa,"a"); panel(axb,"b")
    fig.savefig("figS3.pdf"); plt.close(fig)

# ========================= Fig S4: class-specific calibration resolution =========================
def fig_classres():
    fig,(axa,axb)=plt.subplots(1,2,figsize=(7.3,3.9),gridspec_kw=dict(wspace=0.34))
    cr=AD["classres"]; order=sorted(ORDER,key=lambda m:cr[m]["nN"]); y=np.arange(len(order))
    for i,m in enumerate(order):
        c=cr[m]; col=RC[AD3["reason"][m]]
        axa.plot([1,c["nN"]],[i,i],"-",color="#dddddd",lw=1.5,zorder=1)
        axa.scatter(c["nN"],i,s=48,color=col,edgecolor="white",lw=0.5,zorder=3)
    axa.set_xscale("log"); axa.set_yticks(y); axa.set_yticklabels([SHORT[m] for m in order],fontsize=7)
    axa.set_xlabel("reject-class calibration inputs $n_{\\mathrm{cal},\\mathrm{NO}}$")
    axa.set_title("Reject-class calibration size",fontsize=9.2); axa.grid(axis="x",color=GRID,lw=0.6,which="both")
    axa.set_xlim(2.5,1500); axa.set_ylim(-.6,len(order))
    axa.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor=RC[k],label=k,ms=6.5) for k in ["certified","risk-limited","resolution-limited"]],loc="lower right",fontsize=6.2)
    # panel b: resolution floor (x) vs best achievable CP (y) -- two escalation mechanisms
    axb.axhspan(5,140,color="#b3202c",alpha=0.05,zorder=0); axb.axvspan(5,140,color="#b3202c",alpha=0.05,zorder=0)
    axb.axhline(5,ls="--",lw=1.0,color="#b3202c"); axb.axvline(5,ls="--",lw=1.0,color="#b3202c")
    for m in ORDER:
        axb.scatter(cr[m]["floorN"],AD3["mincpN"][m],s=48,color=RC[AD3["reason"][m]],edgecolor="white",lw=0.5,zorder=4)
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlabel("resolution floor $\\alpha_{\\min,\\mathrm{NO}}$ (%)"); axb.set_ylabel("best achievable CP bound (%)")
    axb.set_title("Two mechanisms of reject-class escalation",fontsize=8.6)
    axb.grid(color=GRID,lw=0.5,which="both"); axb.set_xlim(0.4,140); axb.set_ylim(0.4,140)
    axb.text(0.55,90,"risk-\nlimited",fontsize=6.2,color="#d97706",va="top",fontweight="bold")
    axb.text(12,90,"resolution-\nlimited",fontsize=6.2,color="#b3202c",va="top",fontweight="bold")
    axb.text(0.55,0.85,"certified",fontsize=6.6,color="#2e7d32",fontweight="bold")
    axb.annotate("MMed-3.1",(4.65,8.4),(1.2,20),fontsize=6.0,color="#d97706",arrowprops=dict(arrowstyle="->",lw=0.6,color="#d97706"))
    axb.annotate("OpenBioLLM",(7.31,13.7),(13,30),fontsize=6.0,color="#b3202c",arrowprops=dict(arrowstyle="->",lw=0.6,color="#b3202c"))
    axb.annotate("BioMistral",(82.22,82.2),(20,45),fontsize=6.0,color="#b3202c",arrowprops=dict(arrowstyle="->",lw=0.6,color="#b3202c"))
    axb.legend(handles=[Line2D([0],[0],marker="o",color="w",markerfacecolor=RC[k],label=k,ms=6.5) for k in ["certified","risk-limited","resolution-limited"]],fontsize=6.0,loc="lower right")
    panel(axa,"a"); panel(axb,"b")
    fig.savefig("figS4.pdf",bbox_inches="tight"); plt.close(fig)

# ========================= Fig S5: statistical robustness (delta + cal size) =========================
def fig_robust():
    fig,(axa,axb)=plt.subplots(1,2,figsize=(7.3,3.9),gridspec_kw=dict(wspace=0.32))
    deltas=[0.10,0.05,0.01]; xs=np.arange(3)
    order=sorted(ORDER,key=lambda m:D["delta"][m][0.05])
    for m in order:
        ys=[D["delta"][m][d] for d in deltas]
        axa.plot(xs,ys,"o-",ms=3.4,lw=1.2,color=FAM[fam(m)],alpha=.8)
    axa.axhline(25,ls=":",lw=0.9,color="#666")
    axa.set_xticks(xs); axa.set_xticklabels(["0.10\n(90%)","0.05\n(95%)","0.01\n(99%)"],fontsize=7.5)
    axa.set_xlabel("confidence budget $\\delta$ (conf. $1-\\delta$)"); axa.set_ylabel("marginal frontier size (% BF16)")
    axa.set_title("Frontier vs confidence level",fontsize=9); axa.grid(axis="y",color=GRID,lw=0.6)
    axa.legend(handles=[Line2D([0],[0],color=c,lw=2,label=f) for f,c in FAM.items()],fontsize=6.6,loc="upper left")
    ncs=sorted(D["calsize"]); los=[D["calsize"][n]["lo"] for n in ncs]; his=[D["calsize"][n]["hi"] for n in ncs]
    meds=[D["calsize"][n]["med"] for n in ncs]
    axb.fill_between(ncs,los,his,color="#1f4e79",alpha=0.12,label="frontier range")
    axb.plot(ncs,his,"o-",color="#1f4e79",ms=4,lw=1.4,label="max (least compressible)")
    axb.plot(ncs,meds,"s-",color="#2e7d32",ms=3.5,lw=1.2,label="median")
    axb.plot(ncs,los,"^-",color="#666",ms=3.5,lw=1.0,label="min (4$\\times$ floor)")
    axb.set_xlabel("calibration set size $n_{\\mathrm{cal}}$"); axb.set_ylabel("marginal frontier size (% BF16)")
    axb.set_title("Frontier vs calibration size ($\\alpha=5\\%$)",fontsize=9); axb.grid(color=GRID,lw=0.6)
    axb.legend(fontsize=6.6,loc="upper right"); panel(axa,"a"); panel(axb,"b")
    fig.savefig("figS5.pdf"); plt.close(fig)

# ========================= Fig S6: split / grid stability =========================
def fig_gridstab():
    fig,(axa,axb)=plt.subplots(1,2,figsize=(7.4,3.9),gridspec_kw=dict(wspace=0.30))
    Gs=[25,50,100,200]; xs=list(range(len(Gs)))
    lo=[AD3["grid"][G]["lo"] for G in Gs]; med=[AD3["grid"][G]["med"] for G in Gs]; hi=[AD3["grid"][G]["hi"] for G in Gs]
    axa.fill_between(xs,lo,hi,color="#1f4e79",alpha=0.12,label="frontier range")
    axa.plot(xs,hi,"o-",color="#1f4e79",lw=1.4,ms=4,label="max (least compressible)")
    axa.plot(xs,med,"s-",color="#2e7d32",lw=1.2,ms=3.5,label="median")
    axa.plot(xs,lo,"^-",color="#666",lw=1.0,ms=3.5,label="min ($4\\times$ floor)")
    axa.set_xticks(xs); axa.set_xticklabels(Gs); axa.set_xlabel("threshold grid size $G$")
    axa.set_ylabel("marginal frontier size (% BF16)"); axa.set_ylim(22,40)
    axa.set_title("Threshold-grid sensitivity",fontsize=9.2); axa.grid(color=GRID,lw=0.6); axa.legend(fontsize=6.6,loc="center right")
    order=sorted(ORDER,key=lambda m:np.median(AD3["split_raw"][m]))
    bp=axb.boxplot([AD3["split_raw"][m] for m in order],vert=False,widths=0.62,patch_artist=True,showfliers=False)
    for patch,m in zip(bp["boxes"],order): patch.set_facecolor(FAM[fam(m)]); patch.set_alpha(0.55); patch.set_edgecolor("#333")
    for md in bp["medians"]: md.set_color("#111")
    axb.set_yticks(range(1,len(order)+1)); axb.set_yticklabels([SHORT[m] for m in order],fontsize=7)
    axb.axvline(25,ls=":",lw=0.9,color="#666"); axb.set_xlabel("marginal frontier size (% BF16)")
    axb.set_title("Split stability (20 patient-disjoint splits)",fontsize=9.0); axb.grid(axis="x",color=GRID,lw=0.6)
    panel(axa,"a"); panel(axb,"b")
    fig.savefig("figS6.pdf",bbox_inches="tight"); plt.close(fig)

if __name__=="__main__":
    fig_frontiers(); fig_prevalence(); fig_escalated(); fig_classres(); fig_robust(); fig_gridstab()
    print("wrote figS1.pdf .. figS6.pdf")
