#!/usr/bin/env python3
"""Delete the repos named in public_manifest.json — recovery / re-stage helper.

Use this to remove the partially-uploaded compliant repos (e.g. after hitting the private-storage limit)
so you can re-stage cleanly. Safe by default: lists, and deletes only with --yes.

    python cleanup_public.py                 # dry run: list manifest repos
    python cleanup_public.py --yes           # delete the publish repos
    python cleanup_public.py --include-held --yes
"""
import argparse, json
from huggingface_hub import delete_repo

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="public_manifest.json")
ap.add_argument("--include-held", action="store_true")
ap.add_argument("--yes", action="store_true")
a = ap.parse_args()

m = json.load(open(a.manifest))
repos = [it["repo_id"] for it in m["publish"]]
if a.include_held:
    repos += [h["repo_id"] for h in m.get("held", [])]

print(f"{len(repos)} repos in manifest:")
for r in repos:
    print("  ", r)
if not a.yes:
    print("\nDry run. Re-run with --yes to delete these PERMANENTLY (missing ones are skipped).")
    raise SystemExit(0)

for r in repos:
    try:
        delete_repo(repo_id=r, repo_type="model")
        print("deleted", r)
    except Exception as e:
        print("skip", r, "->", repr(e)[:100])
print(f"\nDone ({len(repos)} attempted).")
