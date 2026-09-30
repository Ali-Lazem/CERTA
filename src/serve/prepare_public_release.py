#!/usr/bin/env python3
"""Build the PUBLIC release layer from the private CERTA tree, WITHOUT moving the baked weights.

Per model it copies the applicable upstream license(s), writes the exact NOTICE + "Built with ..." line,
applies the configured AUP handling (Llama only), ships the weights-aware handler, appends the compliant
spaced model name + attribution to the card, and records a full provenance entry in public_manifest.json.

HuatuoGPT-o1-72B is released under a conservative DUAL-OBLIGATION family "huatuo_qwen": its card declares
Apache-2.0 and its base is Qwen2.5-72B, so each repo carries BOTH LICENSE-APACHE-2.0 and LICENSE-QWEN, the
exact Qwen NOTICE attribution, "Built with Qwen", and an UPSTREAM_LICENSES.md chain. license: other.

Enforced invariant (exits non-zero if violated):
  publish_models=11 held_models=0 | llama3=5 llama3.1=1 llama3.3=1 qwen=1 huatuo_qwen=1 apache=2

Run from the folder containing hf/lazem/CERTA-* (base checkpoints available for the LICENSE copy):
    python prepare_public_release.py --org lazem
Requires LICENSE-APACHE-2.0 present next to this script (shipped in the release folder).
"""
import argparse, glob, json, os, shutil

ORG_DEFAULT = "lazem"
MODELS_DEFAULT = "/scratch/SCWF00175/shared/models"
HERE = os.path.dirname(os.path.abspath(__file__))
HANDLER = os.path.join(HERE, "certa_handler.py")          # weights-aware handler shipped into each repo
APACHE_TPL = os.path.join(HERE, "LICENSE-APACHE-2.0")     # full Apache-2.0 text, for huatuo_qwen repos

# Bundled fallback license texts, used only when a base checkpoint folder ships no LICENSE of its own.
FAMILY_LICENSE_TPL = {"llama3": "LICENSE-LLAMA3", "llama3.1": "LICENSE-LLAMA3.1",
                      "llama3.3": "LICENSE-LLAMA3.3", "apache": "LICENSE-APACHE-2.0"}

CERTA_CONTRIB_LICENSE = "Apache-2.0"
CERTA_COPYRIGHT = "Copyright (c) 2026 Ali H. Lazem and William J. Teahan."
CERTA_LICENSE_HEADER = (
    "CERTA CONTRIBUTION LICENSE\n"
    "==========================\n\n"
    f"{CERTA_COPYRIGHT}\n\n"
    "This license applies ONLY to the CERTA-authored files in this repository:\n"
    "  - certa_handler.py\n  - certa_config.json\n  - certa_thresholds.json\n"
    "  - the CERTA-authored portions of README.md\n\n"
    "It does NOT apply to, and does NOT relicense, the model weights. The weights are a quantized\n"
    "derivative of the upstream base model and remain governed by the upstream model license(s)\n"
    "included in this repository (LICENSE and/or LICENSE-QWEN / LICENSE-APACHE-2.0, as applicable).\n\n"
    "The CERTA-authored files listed above are licensed under the Apache License, Version 2.0,\n"
    "the full text of which follows.\n\n"
    "-------------------------------------------------------------------------------\n\n")


def write_certa_license(repo_dir):
    apache = open(APACHE_TPL).read() if os.path.isfile(APACHE_TPL) else \
        "Obtain the Apache License 2.0 text from https://www.apache.org/licenses/LICENSE-2.0\n"
    open(os.path.join(repo_dir, "LICENSE-CERTA"), "w").write(CERTA_LICENSE_HEADER + apache)

BASE_DIR = {  # display -> local checkpoint dir (authoritative LICENSE copy), from models.txt
 "Qwen2.5-72B":"qwen2.5-72b-instruct", "HuatuoGPT-o1-72B":"huatuogpt-o1-72b", "Med42-70B":"med42-70b",
 "OpenBioLLM-70B":"openbiollm-70b", "Llama-3-70B":"llama3-70b-instruct-v3", "MMed-Llama-3.1-70B":"mmedllama3-70b",
 "Llama-3.3-70B":"llama3-70b", "BioMistral-7B":"biomistral-7b", "Llama-3-8B":"llama3-8b-instruct",
 "Mistral-7B":"mistral-7b-instruct-v0.1", "MMed-Llama-3-8B":"mmed-llama3-8b-enins"}

FAMILY = {
 "Qwen2.5-72B":"qwen", "HuatuoGPT-o1-72B":"huatuo_qwen",
 "Med42-70B":"llama3", "OpenBioLLM-70B":"llama3", "Llama-3-70B":"llama3", "Llama-3-8B":"llama3", "MMed-Llama-3-8B":"llama3",
 "MMed-Llama-3.1-70B":"llama3.1", "Llama-3.3-70B":"llama3.3",
 "BioMistral-7B":"apache", "Mistral-7B":"apache"}

PUBLIC_BASE = {  # display -> compliant PUBLIC repo base (Llama mark leads where required; CERTA kept)
 "Qwen2.5-72B":"CERTA-Qwen2.5-72B", "HuatuoGPT-o1-72B":"CERTA-HuatuoGPT-o1-72B",
 "Med42-70B":"Llama-3-Med42-70B-CERTA", "OpenBioLLM-70B":"Llama-3-OpenBioLLM-70B-CERTA",
 "Llama-3-70B":"Llama-3-70B-CERTA", "Llama-3-8B":"Llama-3-8B-CERTA", "MMed-Llama-3-8B":"Llama-3-MMed-8B-CERTA",
 "MMed-Llama-3.1-70B":"Llama-3.1-MMed-70B-CERTA", "Llama-3.3-70B":"Llama-3.3-70B-CERTA",
 "BioMistral-7B":"CERTA-BioMistral-7B", "Mistral-7B":"CERTA-Mistral-7B"}

ULTIMATE = {
 "Qwen2.5-72B":"Qwen/Qwen2.5-72B-Instruct", "HuatuoGPT-o1-72B":"Qwen/Qwen2.5-72B-Instruct",
 "Med42-70B":"meta-llama/Meta-Llama-3-70B-Instruct", "OpenBioLLM-70B":"meta-llama/Meta-Llama-3-70B-Instruct",
 "Llama-3-70B":"meta-llama/Meta-Llama-3-70B-Instruct", "Llama-3-8B":"meta-llama/Meta-Llama-3-8B-Instruct",
 "MMed-Llama-3-8B":"meta-llama/Meta-Llama-3-8B", "MMed-Llama-3.1-70B":"meta-llama/Llama-3.1-70B",
 "Llama-3.3-70B":"meta-llama/Llama-3.3-70B-Instruct",
 "BioMistral-7B":"mistralai/Mistral-7B-Instruct-v0.1", "Mistral-7B":"mistralai/Mistral-7B-Instruct-v0.1"}

REQUIRED_PREFIX = {"llama3":"Llama 3", "llama3.1":"Llama", "llama3.3":"Llama", "qwen":"", "huatuo_qwen":"", "apache":""}

# license_review_status per model: "verified" = license page/text inspected directly;
# "declared_pending_verification" = reported from the declared tag/family, direct text check outstanding;
# "conservative_dual_obligation" = HuatuoGPT (Apache tag + Qwen base, packaged with both, author clarification pending).
REVIEW_STATUS = {
 "Llama-3-70B":"verified", "Med42-70B":"verified", "OpenBioLLM-70B":"verified",
 "Qwen2.5-72B":"verified", "MMed-Llama-3-8B":"verified",
 "Llama-3-8B":"declared_pending_verification", "Llama-3.3-70B":"declared_pending_verification",
 "MMed-Llama-3.1-70B":"declared_pending_verification", "BioMistral-7B":"declared_pending_verification",
 "Mistral-7B":"declared_pending_verification",
 "HuatuoGPT-o1-72B":"conservative_dual_obligation"}
BUILT = {"llama3":"Built with Meta Llama 3", "llama3.1":"Built with Llama", "llama3.3":"Built with Llama",
         "qwen":"Built with Qwen", "huatuo_qwen":"Built with Qwen", "apache":""}
URLS = {
 "llama3":("https://llama.meta.com/llama3/license","https://llama.meta.com/llama3/use-policy"),
 "llama3.1":("https://llama.meta.com/llama3_1/license","https://llama.meta.com/llama3_1/use-policy"),
 "llama3.3":("https://www.llama.com/llama3_3/license","https://www.llama.com/llama3_3/use-policy"),
 "qwen":("https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE",""),
 "huatuo_qwen":("https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/LICENSE",""),
 "apache":("https://www.apache.org/licenses/LICENSE-2.0","")}

QWEN_ATTR = "Qwen is licensed under the Qwen LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved."


def notice_text(fam, base_id):
    if fam == "llama3":
        return ("Built with Meta Llama 3\n\nMeta Llama 3 is licensed under the Meta Llama 3 Community "
                "License, Copyright (c) Meta Platforms, Inc. All Rights Reserved.\n")
    if fam == "llama3.1":
        return ("Built with Llama\n\nLlama 3.1 is licensed under the Llama 3.1 Community License, "
                "Copyright (c) Meta Platforms, Inc. All Rights Reserved.\n")
    if fam == "llama3.3":
        return ("Built with Llama\n\nLlama 3.3 is licensed under the Llama 3.3 Community License Agreement, "
                "Copyright (c) Meta Platforms, Inc. All Rights Reserved.\n")
    if fam == "qwen":
        return (f"Built with Qwen\n\n{QWEN_ATTR}\n\nQuantized derivative of {base_id}. Retain this notice and "
                f"the included Qwen license. Commercial use is subject to the Qwen license, including its 100M "
                f"monthly-active-users provision (review before any commercial deployment).\n")
    if fam == "huatuo_qwen":
        return (f"Built with Qwen\n\n{QWEN_ATTR}\n\nThis repository is a quantized derivative of {base_id} "
                f"(FreedomIntelligence/HuatuoGPT-o1-72B), which is declared Apache-2.0 by its authors and "
                f"identifies Qwen/Qwen2.5-72B-Instruct as its base model. The applicable Apache-2.0 and Qwen "
                f"licensing, attribution, and notice requirements are retained. Only quantization "
                f"(NF4 / LLM.int8()) was applied by the CERTA pipeline.\n")
    return (f"Quantized derivative of {base_id}, licensed under the Apache License, Version 2.0. "
            f"Only quantization (NF4 / LLM.int8()) was applied; no other modification.\n")


def copy_base_license(base_dir, repo_dir):
    copied = []
    if base_dir and os.path.isdir(base_dir):
        for pat in ("LICENSE*", "USE_POLICY*", "NOTICE*", "Notice*"):
            for f in glob.glob(os.path.join(base_dir, pat)):
                if os.path.isfile(f):
                    shutil.copy(f, os.path.join(repo_dir, os.path.basename(f))); copied.append(os.path.basename(f))
    return copied


def assemble_huatuo(repo_dir, models_dir):
    """Dual-obligation kit: Qwen license (from the Qwen base checkpoint) + Apache-2.0 text + provenance."""
    copied = []
    qdir = os.path.join(models_dir, BASE_DIR["Qwen2.5-72B"])
    qlic = sorted(glob.glob(os.path.join(qdir, "LICENSE*")))
    if qlic:
        shutil.copy(qlic[0], os.path.join(repo_dir, "LICENSE-QWEN")); copied.append("LICENSE-QWEN")
    if os.path.isfile(APACHE_TPL):
        shutil.copy(APACHE_TPL, os.path.join(repo_dir, "LICENSE-APACHE-2.0")); copied.append("LICENSE-APACHE-2.0")
    open(os.path.join(repo_dir, "UPSTREAM_LICENSES.md"), "w").write(
        "Upstream model:\nFreedomIntelligence/HuatuoGPT-o1-72B\n\n"
        "Declared upstream license:\nApache-2.0 (as declared on the upstream Hugging Face model card)\n\n"
        "Declared base model:\nQwen/Qwen2.5-72B-Instruct\n\n"
        "Additional upstream licensing:\nQwen LICENSE AGREEMENT, September 19, 2024\n\n"
        "CERTA modification:\nQuantization (NF4 / LLM.int8()) and CERTA serving/certification metadata.\n")
    gen = os.path.join(repo_dir, "LICENSE")   # drop the generic build-time LICENSE (superseded by the two above)
    if os.path.isfile(gen):
        os.remove(gen)
    return copied


def handle_aup(repo_dir, fam, mode, copied):
    """Explicit aup_handling record; write AUP.md as needed. Only Llama families require an AUP."""
    if not fam.startswith("llama"):
        return None
    _, url = URLS[fam]
    policy_copied = [c for c in copied if c.upper().startswith("USE_POLICY")]
    if mode == "text" and policy_copied:
        return {"mode": "text", "file": policy_copied[0], "reference_url": url}
    open(os.path.join(repo_dir, "AUP.md"), "w").write(
        f"# Acceptable Use Policy\n\nUse of this model is subject to the Meta Llama Acceptable Use Policy.\n"
        f"Official policy: {url}\n")
    return {"mode": "reference", "reference_url": url, "file": "AUP.md"}


def attribution_md(model_name, base_id, fam, copied):
    lic_url, aup_url = URLS[fam]
    out = ["\n## License and attribution\n", f"**Model name:** {model_name}\n"]
    out.append(notice_text(fam, base_id).strip() + "\n")   # notice already carries the "Built with ..." line
    if fam.startswith("llama"):
        out.append(f"Quantized derivative of `{base_id}`, distributed under the Llama Community License. "
                   f"Full license ({lic_url}) and Acceptable Use Policy ({aup_url}) apply; a license copy is "
                   f"included (LICENSE) and the AUP is provided (AUP.md).\n")
    elif fam == "qwen":
        out.append(f"Quantized derivative of `{base_id}`, licensed under the Qwen LICENSE AGREEMENT "
                   f"({lic_url}); a license copy is included. Retain this attribution.\n")
    elif fam == "huatuo_qwen":
        out.append("Upstream licensing: HuatuoGPT-o1-72B is declared Apache-2.0 by FreedomIntelligence and "
                   "identifies Qwen/Qwen2.5-72B-Instruct as its base model. This CERTA quantized derivative "
                   "preserves the applicable Huatuo/Apache and Qwen licensing and attribution requirements "
                   "(see LICENSE-APACHE-2.0, LICENSE-QWEN, NOTICE, UPSTREAM_LICENSES.md). This checkpoint was "
                   "quantized by the CERTA pipeline and has not been fine-tuned by the CERTA authors.\n")
    else:
        out.append(f"Quantized derivative of `{base_id}`, licensed under the Apache License, Version 2.0 "
                   f"({lic_url}); a license copy is included.\n")
    out.append(f"\nUpstream license files included: {', '.join(sorted(set(copied))) if copied else 'NONE — add before publishing'}.\n")
    out.append(f"\n**CERTA contribution license.** The CERTA-authored files (`certa_handler.py`, "
               f"`certa_config.json`, `certa_thresholds.json`, and the CERTA text in this card) are "
               f"{CERTA_COPYRIGHT.rstrip('.')}, licensed under {CERTA_CONTRIB_LICENSE} (see `LICENSE-CERTA`). This does "
               f"not relicense the model weights, which remain under the upstream license(s) above.\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default=ORG_DEFAULT)
    ap.add_argument("--root", default="hf")
    ap.add_argument("--models", default=MODELS_DEFAULT)
    ap.add_argument("--aup-mode", choices=["reference", "text"], default="reference")
    a = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(a.root, a.org, "CERTA-*")))
    if not dirs:
        raise SystemExit(f"No CERTA-*/ under {a.root}/{a.org}/")
    publish, held, held_disp = [], [], set()
    for d in dirs:
        name = os.path.basename(d); body, method = name.rsplit("-", 1); display = body[len("CERTA-"):]
        fam = FAMILY.get(display)
        if fam is None:
            print("?? unknown model, skipping:", name); continue
        if fam == "held" or PUBLIC_BASE.get(display) is None:
            held.append({"repo_id": f"{a.org}/{name}", "release_status": "held", "reason": "held"})
            held_disp.add(display); print("HOLD:", name); continue
        base_id = json.load(open(os.path.join(d, "certa_config.json")))["base_model"]
        copied = assemble_huatuo(d, a.models) if fam == "huatuo_qwen" else copy_base_license(os.path.join(a.models, BASE_DIR[display]), d)
        license_source = "base_checkpoint" if copied else None
        if fam == "huatuo_qwen":
            license_source = "qwen_base + apache_bundled"
        elif not copied and fam in FAMILY_LICENSE_TPL:   # checkpoint had no LICENSE -> bundled fallback
            tpl = os.path.join(HERE, FAMILY_LICENSE_TPL[fam])
            if os.path.isfile(tpl):
                shutil.copy(tpl, os.path.join(d, "LICENSE")); copied = ["LICENSE"]; license_source = "bundled_template"
        open(os.path.join(d, "NOTICE"), "w").write(notice_text(fam, base_id))
        aup = handle_aup(d, fam, a.aup_mode, copied)
        if os.path.isfile(HANDLER):
            shutil.copy(HANDLER, os.path.join(d, "certa_handler.py"))
        write_certa_license(d)   # LICENSE-CERTA: Apache-2.0 for the CERTA-authored files (not the weights)
        public_base = PUBLIC_BASE[display]
        model_name = f"{public_base}-{method}".replace("-", " ")
        # Idempotent: strip any previously-appended attribution block, then write exactly one.
        readme = os.path.join(d, "README.md")
        body = open(readme).read() if os.path.isfile(readme) else ""
        cut = body.find("## License and attribution")
        if cut != -1:
            body = body[:cut].rstrip() + "\n"
        open(readme, "w").write(body + attribution_md(model_name, base_id, fam, copied))
        entry = {
            "repo_id": f"{a.org}/{public_base}-{method}", "folder": d,
            "base_model": base_id, "ultimate_upstream": ULTIMATE[display], "method": method,
            "license_family": fam, "required_name_prefix": REQUIRED_PREFIX[fam], "model_name": model_name,
            "distribution_status": "publish",
            "license_review_status": REVIEW_STATUS[display],
            "license_metadata_inspected": bool(copied),
            "author_clarification": {"requested": False, "received": False},
            "certa_contribution_license": CERTA_CONTRIB_LICENSE,
            "license_files": copied, "license_source": license_source, "aup_handling": aup,
            "weights_included": bool(glob.glob(os.path.join(d, "*.safetensors"))), "_display": display}
        if fam == "huatuo_qwen":
            entry["upstream_declared_license"] = "apache-2.0"
            entry["base_license_family"] = "qwen"
        publish.append(entry)
        print(f"OK  {name}  ->  {a.org}/{public_base}-{method}  [{fam}]  license:{copied or 'ADD'}")

    json.dump({"org": a.org, "publish": publish, "held": held}, open("public_manifest.json", "w"), indent=2)

    # ---- per-generation invariant (counted per distinct MODEL) ----
    model_fam = {it["_display"]: it["license_family"] for it in publish}
    models = set(model_fam)
    fams = ("llama3", "llama3.1", "llama3.3", "qwen", "huatuo_qwen", "apache")
    c = {k: sum(1 for f in model_fam.values() if f == k) for k in fams}
    print(f"\nInvariant: publish_models={len(models)} held_models={len(held_disp)} | " +
          " ".join(f"{k}={c[k]}" for k in fams))
    ok = (len(models) == 11 and len(held_disp) == 0 and c["llama3"] == 5 and c["llama3.1"] == 1
          and c["llama3.3"] == 1 and c["qwen"] == 1 and c["huatuo_qwen"] == 1 and c["apache"] == 2)
    if not ok:
        raise SystemExit("INVARIANT FAILED — expected publish=11, held=0, llama3=5, llama3.1=1, "
                         "llama3.3=1, qwen=1, huatuo_qwen=1, apache=2. Fix before any public upload.")
    print("Invariant OK. Wrote public_manifest.json. Next: python upload_public.py  (private default; --public to release)")


if __name__ == "__main__":
    main()
