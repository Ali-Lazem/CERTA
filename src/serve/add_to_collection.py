#!/usr/bin/env python3
"""Create ONE collection from public_manifest.json and add every publish repo (skips held HuatuoGPT).

Manifest-driven: uses the license-compliant repo names recorded by prepare_public_release.py, so it stays
correct after the rename. Order: NF4 flagship -> LLM.int8() -> RTN evaluation variants; models within a
tier follow the paper's Table 1 order. Private by default. Idempotent: it deletes any prior collection with
the same title first (the old one pointed at the renamed/deleted CERTA-* repos), then rebuilds cleanly.

    hf auth login
    python add_to_collection.py                 # PRIVATE collection
    python add_to_collection.py --public         # public collection
Run after the repos exist on the Hub (upload_public.py).
"""
import argparse, json
from huggingface_hub import HfApi, create_collection, add_collection_item, delete_collection

TITLE = "CERTA: Certified Compression for Clinical Decisions"        # <=60 chars (HF limit)
DESC  = ("CERTA certify-or-escalate layers (no base weights) for quantized clinical "
         "verifiers: NF4 flagship, then LLM.int8(), then RTN variants.")             # <=150 chars

TIER = {"NF4": (0, "Flagship: NF4 4-bit, primary deployable frontier"),
        "LLMINT8": (1, "Deployable: LLM.int8() 8-bit (where the route loads)"),
        "RTN4": (2, "Evaluation variant: round-to-nearest 4-bit (reproducibility, not deployable)"),
        "RTN6": (3, "Evaluation variant: round-to-nearest 6-bit (reproducibility, not deployable)"),
        "RTN8": (4, "Evaluation variant: round-to-nearest 8-bit (reproducibility, not deployable)")}
MODEL_ORDER = ["Qwen2.5-72B", "HuatuoGPT-o1-72B", "Med42-70B", "OpenBioLLM-70B", "Llama-3-70B",
               "MMed-Llama-3.1-70B", "Llama-3.3-70B", "BioMistral-7B", "Llama-3-8B", "Mistral-7B",
               "MMed-Llama-3-8B"]


def method_of(repo_id):
    return repo_id.rsplit("-", 1)[1]


def sort_key(it):
    disp = it.get("_display", "")
    mi = MODEL_ORDER.index(disp) if disp in MODEL_ORDER else 99
    return (TIER.get(method_of(it["repo_id"]), (9, ""))[0], mi, it["repo_id"])


ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="public_manifest.json")
ap.add_argument("--public", action="store_true", help="make the collection public (default: PRIVATE)")
a = ap.parse_args()

m = json.load(open(a.manifest))
org = m["org"]
items = sorted(m["publish"], key=sort_key)
api = HfApi()

# delete any stale collection with the same title (it referenced the old CERTA-* names)
try:
    for c in api.list_collections(owner=org):
        if c.title == TITLE:
            delete_collection(c.slug, missing_ok=True)
            print("deleted stale collection:", c.slug)
except Exception as e:
    print("note: could not list/delete prior collections:", repr(e)[:120])

col = create_collection(title=TITLE, namespace=org, description=DESC, private=not a.public, exists_ok=True)
print(("PUBLIC " if a.public else "private ") + f"collection: {col.slug}")

added = 0
for it in items:
    note = TIER.get(method_of(it["repo_id"]), (9, "CERTA artifact"))[1]
    try:
        add_collection_item(col.slug, item_id=it["repo_id"], item_type="model", note=note, exists_ok=True)
        print("  +", it["repo_id"]); added += 1
    except Exception as e:
        print("  ! skip", it["repo_id"], "->", repr(e)[:100])

if m.get("held"):
    print("held (intentionally NOT added):")
    for h in m["held"]:
        print("   -", h["repo_id"])

print(f"\n{added}/{len(items)} items in collection {col.slug}")
print(f"View: https://huggingface.co/collections/{col.slug}")
