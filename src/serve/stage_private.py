#!/usr/bin/env python3
"""Stage the CERTA Hugging Face repos as PRIVATE with metadata only (no weights) from a laptop.
Uses the version-stable huggingface_hub Python API (works on hub 0.x/1.x/2.x) so it does not depend on
the CLI name or on transformers/safetensors/torch being importable.

Run from the extracted certa-hf-collection folder (the one containing AliLazem/ and MANIFEST.json):
    hf auth login                 # log in first with your (new) write token
    python stage_private.py                 # stage ALL repos, private
    python stage_private.py --only NF4      # only the NF4 repos
    python stage_private.py --only LLMINT8  # only the INT8 repos
"""
import os, glob, argparse
from huggingface_hub import create_repo, upload_folder

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".", help="folder containing <org>/CERTA-*/ dirs")
ap.add_argument("--only", default="", help="only repos whose name ends with this method, e.g. NF4")
ap.add_argument("--public", action="store_true", help="make repos public (default: PRIVATE)")
a = ap.parse_args()

dirs = sorted(glob.glob(os.path.join(a.root, "*", "CERTA-*")))
if not dirs:
    raise SystemExit("No CERTA-*/ dirs found. Run this from inside the extracted certa-hf-collection folder.")
done = 0
for d in dirs:
    if a.only and not os.path.basename(d).endswith("-" + a.only):
        continue
    org = os.path.basename(os.path.dirname(d)); name = os.path.basename(d); repo = f"{org}/{name}"
    create_repo(repo, repo_type="model", private=not a.public, exist_ok=True)
    upload_folder(repo_id=repo, folder_path=d, repo_type="model",
                  commit_message="CERTA layer (metadata; weights added later)")
    print(("PUBLIC " if a.public else "private ") + repo); done += 1
print(f"\nStaged {done} repo(s). Metadata is up; add weights later on the GPU box with quantize_and_push.py.")
