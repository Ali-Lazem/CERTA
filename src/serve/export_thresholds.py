#!/usr/bin/env python3
"""Export CERTA serving thresholds (one JSON per model) from the operational (full-calibration)
certificate, for the deployable quantizers (NF4 4-bit, INT8 8-bit) at alpha in {1%,5%,10%}.
Emits marginal and class-conditional (Mondrian) thresholds plus coverage, Clopper-Pearson bounds,
escalation and amortized size. Numbers come from certa_core (the single source of truth); nothing
is hand-entered. Run from the repository root: python src/serve/export_thresholds.py"""
import json, os, math, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # find certa_core in src/
import certa_core as C
from certa_core import ORDER, marg, mond

HFID = {
    "qwen2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "huatuogpt-o1-72b": "FreedomIntelligence/HuatuoGPT-o1-72B",
    "med42-70b": "m42-health/Llama3-Med42-70B",
    "openbiollm-70b": "aaditya/Llama3-OpenBioLLM-70B",
    "llama3-70b-base": "meta-llama/Meta-Llama-3-70B",
    "mmed-3.1-70b": "Henrychur/MMed-Llama3.1-70B",
    "llama3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "biomistral-7b": "BioMistral/BioMistral-7B",
    "llama3-8b": "meta-llama/Meta-Llama-3-8B",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mmed-8b": "Henrychur/MMed-Llama-3-8B",
}
ALPHAS = [0.01, 0.05, 0.10]
# method -> (quantizer, bits, deployable). NF4/INT8 reduce memory (deployable); RTN retains bfloat16
# storage and is an evaluation/precision-sweep variant, not a memory-reducing deployment.
METHODS = {"nf4": ("bnb", 4, True), "int8": ("bnb", 8, True),
           "rtn4": ("rtn", 4, False), "rtn6": ("rtn", 6, False), "rtn8": ("rtn", 8, False)}
def fin(x):
    return None if (x is None or (isinstance(x, float) and not math.isfinite(x))) else round(float(x), 4)

OUT = os.path.join(os.path.dirname(__file__), "thresholds")
os.makedirs(OUT, exist_ok=True)
for m in ORDER:
    rec = {
        "model": m,
        "base_model": HFID[m],
        "certa_version": "1.0",
        "task": "clinical accept/reject verification decision; thresholds calibrated on PMC-Patients",
        "margin_definition": "abs(logit(YES) - logit(NO)) at the answer position",
        "answer_tokens": {"YES": "YES", "NO": "NO"},
        "serve_rule": "serve the quantized decision if margin >= tau for its predicted class, else escalate to the full-precision reference",
        "certificate": {"type": "full-calibration operational", "delta": C.DELTA, "grid": C.GRID,
                        "n_cal_decisions": int(len(C.CAL)), "n_patients_total": 2484},
        "IMPORTANT": ("The certificate holds ONLY for this calibrated decision distribution; re-calibrate "
                      "on your own data before relying on it for a different task or population. NF4 and INT8 "
                      "reduce memory; round-to-nearest variants retain bfloat16 storage and are not memory-reducing. "
                      "Escalated decisions require the full-precision reference to be reachable."),
        "configs": {},
    }
    for qname, (q, bit, deployable) in METHODS.items():
        cfg = {"quantizer": q, "bits": bit, "deployable": deployable, "alpha": {}}
        if qname == "int8":
            cfg["note"] = "LLM.int8(); fails to load on some BF16-native 70B checkpoints (0% certified coverage there)"
        elif q == "rtn":
            cfg["note"] = "round-to-nearest evaluation variant; retains bfloat16 storage, NOT memory-reducing"
        for a in ALPHAS:
            mg = marg(m, q, bit, alpha=a)
            md = mond(m, q, bit, alpha=a)
            cfg["alpha"][f"{a:.2f}"] = {
                "marginal": {"tau": fin(mg["tau"]), "certified_coverage_pct": fin(mg["certCov"]),
                             "escalation_pct": fin(100 - mg["certCov"]), "cp_upper_pct": fin(mg["cp"]),
                             "amortized_size_pct": fin(mg["size"]),
                             "escalate_all": not math.isfinite(mg["tau"])},
                "class_conditional": {"tau_yes": fin(md["tauY"]), "tau_no": fin(md["tauN"]),
                                      "cov_yes_pct": fin(md["covY"]), "cov_no_pct": fin(md["covN"]),
                                      "cp_yes_pct": fin(md["cpY"]), "cp_no_pct": fin(md["cpN"]),
                                      "escalation_pct": fin(md["esc"]), "amortized_size_pct": fin(md["size"]),
                                      "escalate_all_yes": not math.isfinite(md["tauY"]),
                                      "escalate_all_no": not math.isfinite(md["tauN"])},
            }
        rec["configs"][qname] = cfg
    json.dump(rec, open(os.path.join(OUT, f"{m}.json"), "w"), indent=2)
    q5 = rec["configs"]["nf4"]["alpha"]["0.05"]["class_conditional"]
    print(f"{m:16} NF4 @5%: tauY={q5['tau_yes']} tauN={q5['tau_no']} esc={q5['escalation_pct']}% size={q5['amortized_size_pct']}%")
print(f"\nwrote {len(ORDER)} threshold files to {OUT}/")
