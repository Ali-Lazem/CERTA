# src/serve/ — CERTA Hugging Face release

Packages the CERTA certificate as a Hugging Face **collection**: one repository per (model, compression
method), each carrying the CERTA layer (config, calibrated per-class thresholds, weights-aware serving
handler) and **license-compliant packaging**. Base-model weights are not re-hosted; released quantized
checkpoints are derivative works.

## The collection

53 repositories = **11 NF4** + **9 LLM.int8()** + **33 round-to-nearest** (RTN 8/6/4), across 11 models.
LLM.int8() is omitted for the two BF16-native 70B checkpoints where it collapses to 0% certified coverage
(Llama-3-70B, Med42-70B), matching the paper.

Naming: Llama-derived repositories lead with the Llama mark (e.g. `Llama-3-70B-CERTA-NF4`,
`Llama-3.1-MMed-70B-CERTA-NF4`), as required by the Meta Llama Community License; the non-Llama repositories
use `CERTA-<model>-<method>`.

## Files

| File | Role |
|------|------|
| `export_thresholds.py` | Regenerate `thresholds/<model>.json` (all five methods, alpha 1/5/10%, marginal + per-class) from `certa_core`. |
| `build_hf_repos.py` | Generate one repo directory per (model, method) with `certa_config.json`, `certa_thresholds.json`, model card, handler and a base LICENSE note. |
| `prepare_public_release.py` | Assemble the license kit (upstream license(s), NOTICE, AUP, `LICENSE-CERTA`), write license-compliant repo names, ship the weights-aware handler, and emit `public_manifest.json` with full provenance. Enforces the release invariant. |
| `upload_public.py` | Upload from the manifest. Private by default; `--wrapper-only` uploads metadata without weights; `--public` flips visibility and refuses any repo lacking packaged license text or still under review. |
| `add_to_collection.py` | Create/rebuild the collection from the manifest, ordered NF4 → LLM.int8() → RTN. |
| `quantize_and_push.py` | GPU-side: produce a repo's quantized weights from the exact local checkpoint (optionally upload). |
| `stage_private.py` | Version-stable private staging of the repos via the `huggingface_hub` API. |
| `cleanup_public.py` | Delete the manifest's repositories (recovery / re-stage). |
| `certa_handler.py` | Weights-aware certify-or-escalate handler; loads baked weights if present, else quantizes the base at load. |
| `thresholds/<model>.json` | Calibrated thresholds; machine-derived by `export_thresholds.py`. |
| `licenses/` | Bundled upstream license texts (Apache-2.0, Meta Llama 3 / 3.1 / 3.3) used when a checkpoint ships no LICENSE. |
| `MODEL_CARD_TEMPLATE.md`, `COLLECTION.md`, `MANIFEST.json` | Reference card, collection description, and generated manifest. |

## Release invariant (per generation)

`prepare_public_release.py` and `upload_public.py` fail loudly unless:

```
publish_models = 11 ; held_models = 0
llama3 = 5 ; llama3.1 = 1 ; llama3.3 = 1 ; qwen = 1 ; huatuo_qwen = 1 ; apache = 2
```

## Licensing model

Each repository ships the applicable **upstream weights license** (copied from the base checkpoint, or a
bundled fallback from `licenses/`) plus **`LICENSE-CERTA`** (Apache-2.0) covering only the CERTA-authored
files. HuatuoGPT-o1-72B is packaged conservatively under **both** Apache-2.0 (its declared license) and the
Qwen license (its Qwen2.5-72B base), with an `UPSTREAM_LICENSES.md` recording the chain. `public_manifest.json`
records, per repository: `base_model`, `ultimate_upstream`, `license_family`, `license_source`,
`license_files`, `distribution_status`, `license_review_status`, `author_clarification`, and
`certa_contribution_license`.

## Typical run (from this folder, after `hf auth login`)

```bash
python prepare_public_release.py --org lazem       # license kit + manifest + invariant
python upload_public.py --wrapper-only             # PRIVATE staging (no weights)
python add_to_collection.py                        # build the collection
# at publication, after license-text checks:
python upload_public.py --public --only NF4        # publish weights (public storage, not the private quota)
```

GPU-side weight baking is separate; see `../../slurm/`.
