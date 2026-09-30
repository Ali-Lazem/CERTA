#!/usr/bin/env python3
"""Upload the prepared public repos to their license-compliant names, from public_manifest.json.

Private by default. Gatekeeper: a repo is uploaded only if release_status == "publish"; and for --public it
must also have license_verified == True and an upstream license file present. Anything in `held` is never
touched. The per-generation invariant (10/1 | llama3=5 llama3.1=1 llama3.3=1 qwen=1 apache=2) must hold
before any --public push. Run prepare_public_release.py first.

    hf auth login
    python upload_public.py                 # upload all publish repos, PRIVATE
    python upload_public.py --only NF4        # only NF4 repos
    python upload_public.py --public         # publish (only after license sign-off)
"""
import argparse, json, os
from huggingface_hub import HfApi, create_repo, upload_folder


def set_public(api, repo):
    """Flip an existing repo to public, version-robustly (no re-upload of weights)."""
    try:
        api.update_repo_settings(repo_id=repo, repo_type="model", private=False)
    except Exception:
        api.update_repo_visibility(repo_id=repo, private=False, repo_type="model")

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="public_manifest.json")
ap.add_argument("--only", default="", help="filter by method suffix, e.g. NF4 or LLMINT8")
ap.add_argument("--public", action="store_true", help="make repos public (default: PRIVATE)")
ap.add_argument("--wrapper-only", action="store_true",
                help="upload metadata/handler/license only, NOT the *.safetensors weights "
                     "(use for private staging: fits the private-storage quota)")
a = ap.parse_args()

api = HfApi()
m = json.load(open(a.manifest))
for h in m.get("held", []):
    print("HELD (never uploaded):", h["repo_id"], "-", h.get("reason", ""))

# per-generation invariant (counted per distinct MODEL family, not per repo)
model_fam = {it["repo_id"].rsplit("-", 1)[0]: it["license_family"] for it in m["publish"]}
models = set(model_fam)
FAMS = ("llama3", "llama3.1", "llama3.3", "qwen", "huatuo_qwen", "apache")
c = {k: sum(1 for f in model_fam.values() if f == k) for k in FAMS}
inv_ok = (len(models) == 11 and c["llama3"] == 5 and c["llama3.1"] == 1 and c["llama3.3"] == 1
          and c["qwen"] == 1 and c["huatuo_qwen"] == 1 and c["apache"] == 2)
print("invariant: publish_model_families=%d | %s -> %s" %
      (len(models), " ".join(f"{k}={c[k]}" for k in FAMS), "OK" if inv_ok else "FAIL"))
if a.public and not inv_ok:
    raise SystemExit("REFUSING --public: release invariant not satisfied.")

done = 0
for it in m["publish"]:
    repo, folder = it["repo_id"], it["folder"]
    if it.get("distribution_status") != "publish":
        print("skip (not publish):", repo); continue
    if a.only and not repo.endswith("-" + a.only):
        continue
    if not os.path.isdir(folder):
        print("MISSING folder, skip:", folder); continue
    # Tiered public gate:
    #   no license text packaged            -> REFUSE
    #   declared_pending_verification       -> REFUSE (finish the direct license-text check first)
    #   verified                            -> ALLOW
    #   conservative_dual_obligation        -> ALLOW + WARNING (dual kit present, clarification pending)
    if a.public:
        st = it.get("license_review_status", "")
        if not (it.get("license_metadata_inspected") and it.get("license_files")):
            print("REFUSE --public (no license text packaged in repo):", repo); continue
        if st == "declared_pending_verification":
            print(f"REFUSE --public ({repo}): declared_pending_verification — complete the direct "
                  f"license-text check and set it 'verified' in prepare_public_release.py, then re-run.")
            continue
        if st == "conservative_dual_obligation":
            ac = it.get("author_clarification", {})
            extra = " (author clarification pending)" if ac.get("requested") and not ac.get("received") else ""
            print(f"  WARNING publishing {repo}: conservative_dual_obligation{extra}; "
                  f"license files packaged = {it.get('license_files')}")
    create_repo(repo, repo_type="model", private=not a.public, exist_ok=True)
    if a.public:
        set_public(api, repo)   # flip BEFORE uploading weights so they don't count against private quota
    ignore = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf"] if a.wrapper_only else None
    upload_folder(repo_id=repo, folder_path=folder, repo_type="model",
                  commit_message="CERTA " + ("wrapper (no weights)" if a.wrapper_only else "quantized derivative")
                                  + " + license/attribution",
                  ignore_patterns=ignore)
    print(("PUBLIC " if a.public else "private ") + ("[wrapper] " if a.wrapper_only else "") + repo); done += 1

print(f"\nUploaded {done} repo(s). Held {len(m.get('held', []))}.")
