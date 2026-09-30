#!/usr/bin/env python3
"""Generate one Hugging Face repository directory per (model, compression method) following the
naming scheme CERTA-<model>-<METHOD>, gathered under a single collection. Each directory carries the
model card, certa_config.json, the method's certa_thresholds.json, the handler and a LICENSE note.
The quantized weights are NOT produced here (no GPU/weights); quantize_and_push.py adds them and uploads.

Categories:
  deployable  : NF4, LLMINT8   (memory-reducing)
  evaluation  : RTN4, RTN6, RTN8  (retain bfloat16 storage; reproducibility variants, not deployment)
INT8 is skipped for models where LLM.int8() does not load (0% certified coverage).

Run from the repo root: python build_hf_repos.py [ORG]   (default ORG=lazem)
"""
import json, os, glob, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORG = sys.argv[1] if len(sys.argv) > 1 else "lazem"
OUT = os.path.join(HERE, "hf")
NAME = {"qwen2.5-72b": "Qwen2.5-72B", "huatuogpt-o1-72b": "HuatuoGPT-o1-72B", "med42-70b": "Med42-70B",
        "openbiollm-70b": "OpenBioLLM-70B", "llama3-70b-base": "Llama-3-70B", "mmed-3.1-70b": "MMed-Llama-3.1-70B",
        "llama3.3-70b": "Llama-3.3-70B", "biomistral-7b": "BioMistral-7B", "llama3-8b": "Llama-3-8B",
        "mistral-7b": "Mistral-7B", "mmed-8b": "MMed-Llama-3-8B"}
LIC = {"Qwen/Qwen2.5-72B-Instruct": "other", "FreedomIntelligence/HuatuoGPT-o1-72B": "other",
       "m42-health/Llama3-Med42-70B": "llama3", "aaditya/Llama3-OpenBioLLM-70B": "llama3",
       "meta-llama/Meta-Llama-3-70B-Instruct": "llama3", "Henrychur/MMed-Llama3.1-70B": "llama3.1",
       "meta-llama/Llama-3.3-70B-Instruct": "llama3.3", "BioMistral/BioMistral-7B": "apache-2.0",
       "meta-llama/Meta-Llama-3-8B-Instruct": "llama3", "mistralai/Mistral-7B-Instruct-v0.1": "apache-2.0",
       "Henrychur/MMed-Llama-3-8B-EnIns": "llama3"}
LABEL = {"nf4": "NF4", "int8": "LLMINT8", "rtn4": "RTN4", "rtn6": "RTN6", "rtn8": "RTN8"}
RECIPE = {
    "nf4": {"library": "bitsandbytes", "load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "bfloat16"},
    "int8": {"library": "bitsandbytes", "load_in_8bit": True},
    "rtn4": {"method": "round-to-nearest", "bits": 4, "group_size": 128, "clip_search": [0.7, 0.8, 0.9, 1.0], "storage": "bfloat16"},
    "rtn6": {"method": "round-to-nearest", "bits": 6, "group_size": 128, "clip_search": [0.7, 0.8, 0.9, 1.0], "storage": "bfloat16"},
    "rtn8": {"method": "round-to-nearest", "bits": 8, "group_size": 128, "clip_search": [0.7, 0.8, 0.9, 1.0], "storage": "bfloat16"},
}
HANDLER = os.path.join(HERE, "certa_handler.py")


def usable_int8(cfg):
    a = cfg["alpha"]["0.10"]["marginal"]["certified_coverage_pct"]
    return a is not None and a > 0


def card(m, meth, base, deployable, th_block):
    cc = th_block["alpha"]["0.05"]["class_conditional"]
    esc = "n/a" if cc["escalation_pct"] is None else f"{cc['escalation_pct']:.1f}"
    size = "n/a" if cc["amortized_size_pct"] is None else f"{cc['amortized_size_pct']:.1f}"
    lic = LIC.get(base, "other")
    if deployable:
        kind = "**Deployable** (memory-reducing quantizer)."
        tier = ("This is a **primary NF4 deployable artifact**." if meth == "nf4"
                else "This is a **deployable LLM.int8() artifact** (published where the 8-bit route loads).")
        runblock = f"""## Hardware
The wrapper is lightweight, but it does not remove the compute needed to run the model. At load time it
downloads the base model `{base}` and quantizes it with bitsandbytes, so a CUDA GPU with enough memory to
hold the {LABEL[meth]} model is required for inference. The wrapper only removes the need to re-host a
separate copy of the (quantized) weights.

## Usage
```python
from certa_handler import EndpointHandler
h = EndpointHandler(".")                       # downloads + quantizes {base} on a CUDA GPU
out = h({{"inputs": "<YES/NO clinical-verifier prompt>", "alpha": 0.05}})
# escalate=True -> refer this decision to the full-precision (BF16) reference model
```"""
    else:
        kind = ("**Evaluation variant** (round-to-nearest; retains bfloat16 storage, not memory-reducing, "
                "and not runnable through the load-time wrapper).")
        tier = ("This is an **evaluation / reproducibility variant** for the paper's precision sweep, not a "
                "deployable configuration.")
        runblock = """## Usage
This variant carries the calibrated thresholds so the paper's precision sweep can be reproduced, but it is
round-to-nearest at bfloat16 storage, so it is **not** memory-reducing and is **not** served by the
wrapper. Use an NF4 (or, where available, LLM.int8()) repository in this collection for deployment."""
    return f"""---
license: {lic}   # VERIFY against the base model's current license before any public release
base_model: {base}
tags: [certa, certified-compression, clinical-nlp, selective-prediction, quantization]
---

# CERTA-{NAME[m]}-{LABEL[meth]}

**This repository does not contain the original model weights.** It provides the CERTA certification
configuration and serving wrapper for the referenced base model `{base}`. The certificate is calibrated
for the clinical accept/reject decision task and evaluation distribution described in the accompanying
paper; deployment on a different task or population requires recalibration.

CERTA is a decision-level certify-or-escalate layer around a quantized model: a decision is served by the
quantized model only when its margin clears the calibrated threshold for its predicted class at the
requested risk level, and is otherwise escalated to the full-precision (BF16) reference. This repository
does not fine-tune or modify the base model.

**Compression method:** {LABEL[meth]}. {kind} {tier}

## What CERTA certifies (and what it does not)
- **Certified:** compression-induced disagreement with the BF16 reference on served decisions, bounded at
  risk level `alpha` with confidence `1 - delta` (delta = 0.10), for the clinical accept/reject
  verification decision calibrated on PMC-Patients.
- **Not certified:** clinical correctness of the reference decision; and any guarantee on a **different
  task or population** (re-calibrate with `export_thresholds.py`).
- The thresholds in `certa_thresholds.json` are the exact calibrated values from the study, not
  re-estimated here.

## Class-conditional operating point (alpha = 5%)
Escalation {esc}% ; amortized size {size}% of the BF16 reference.
Full thresholds (alpha = 1/5/10%, marginal and per-class) are in `certa_thresholds.json`.

{runblock}

## Citation
Lazem & Teahan, *Do We Need the Largest LLMs for Every Clinical Decision?* See the CERTA source repository.
"""


rows = []
for f in sorted(glob.glob(os.path.join(HERE, "thresholds", "*.json"))):
    d = json.load(open(f)); m = d["model"]; base = d["base_model"]
    for meth, block in d["configs"].items():
        if meth == "int8" and not usable_int8(block):
            rows.append((m, base, meth, "skipped (INT8 fails to load)")); continue
        deployable = block["deployable"]
        repo = f"CERTA-{NAME[m]}-{LABEL[meth]}"
        rdir = os.path.join(OUT, ORG, repo); os.makedirs(rdir, exist_ok=True)
        json.dump({"repo": f"{ORG}/{repo}", "base_model": base, "certa_method": LABEL[meth],
                   "quantizer": block["quantizer"], "bits": block["bits"], "deployable": deployable,
                   "category": "deployable" if deployable else "evaluation",
                   "quantization": RECIPE[meth], "task": d["task"], "answer_tokens": d["answer_tokens"],
                   "alpha_options": [0.01, 0.05, 0.10], "default_alpha": 0.05,
                   "certa_version": d["certa_version"]},
                  open(os.path.join(rdir, "certa_config.json"), "w"), indent=2)
        thr = {"method": LABEL[meth], "base_model": base, "answer_tokens": d["answer_tokens"],
               "margin_definition": d["margin_definition"], "IMPORTANT": d["IMPORTANT"],
               "certificate": d["certificate"], "alpha": block["alpha"]}
        json.dump(thr, open(os.path.join(rdir, "certa_thresholds.json"), "w"), indent=2)
        open(os.path.join(rdir, "README.md"), "w").write(card(m, meth, base, deployable, block))
        shutil.copy(HANDLER, os.path.join(rdir, "certa_handler.py"))
        open(os.path.join(rdir, "LICENSE"), "w").write(
            f"This model is a quantized derivative of {base}.\nIt inherits and is subject to that model's "
            f"license ({LIC.get(base,'see base model')}); obtain and include the full license text and "
            f"comply with all naming/use requirements before publishing. VERIFY before upload.\n")
        rows.append((m, base, meth, f"{ORG}/{repo}  [{'deployable' if deployable else 'evaluation'}]"))

manifest = {"collection": "CERTA: Certified Compression for Clinical Decision Substitution", "org": ORG,
            "repos": [r[3].split()[0] for r in rows if r[3].startswith(ORG)],
            "skipped": [{"model": r[0], "method": r[2]} for r in rows if "skipped" in r[3]]}
json.dump(manifest, open(os.path.join(OUT, "MANIFEST.json"), "w"), indent=2)
n_dep = sum(1 for r in rows if "deployable" in r[3]); n_eval = sum(1 for r in rows if "evaluation" in r[3])
open(os.path.join(OUT, "COLLECTION.md"), "w").write(f"""# CERTA: Certified Compression for Clinical Decision Substitution

A Hugging Face collection of clinical-verifier CERTA artifacts. Each repository provides the CERTA
certify-or-escalate layer (calibrated thresholds, config, handler); **none of them re-host the base model
weights**. Group these repositories into one collection named above under the **{ORG}** namespace.

Naming: `CERTA-<model>-<METHOD>`, where METHOD is the actual compression scheme (NF4, LLMINT8, RTN4/6/8),
so the method is explicit rather than a bare "4-bit"/"8-bit" label.

The repositories are **not all equally important** -- lead the collection with NF4:

- **Flagship -- NF4 (11 repos).** The primary deployable 4-bit result and the paper's main frontier.
- **Deployable -- LLM.int8() (9 repos).** Published where the 8-bit route loads; skipped on the two
  BF16-native 70B checkpoints where it collapses to 0% certified coverage, as reported in the paper.
- **Evaluation variants -- RTN4/RTN6/RTN8 (33 repos).** Round-to-nearest at bfloat16 storage; published
  for reproducibility of the precision sweep only, not memory-reducing and not deployable.

Release stage: **wrapper now, weights later.** These wrapper artifacts load and quantize the base model at
runtime; pre-quantized NF4/INT8 weights can be added to the same repositories in a second pass once each
base model's license has been verified for public redistribution.

Each repository references its named base model plus the CERTA layer; it does not fine-tune the base model.
The certificate is conditional on the clinical accept/reject task, the PMC-Patients calibration
distribution, alpha, the thresholding procedure and the reference-model definition.

Total generated: {n_dep + n_eval} repositories ({n_dep} deployable, {n_eval} evaluation). See MANIFEST.json.
""")
for r in rows:
    print(f"  {r[3]}")
print(f"\n{n_dep} deployable + {n_eval} evaluation repos under {OUT}/{ORG}/ ; INT8 skipped for the 2 non-loading models.")
