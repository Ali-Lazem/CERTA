# CERTA HuggingFace release — run sheet (wrapper-only, private)

Everything here runs from your **laptop** (no GPU needed for the wrapper release). You are already
logged in as `lazem`. Run every command from inside this extracted folder.

Namespace: `lazem`. Contents already built under `hf/lazem/` (53 repos). No base-model weights are
re-hosted; each repo is the CERTA layer only (handler + thresholds + config + card + LICENSE).

## 0. Prerequisites (once)
```bash
pip install -U "huggingface_hub>=0.24"
hf auth login          # paste a WRITE token; do not paste tokens anywhere else
```

## 1. Delete the 53 old repos (clean slate)
```bash
python delete_certa_repos.py          # dry run: lists the CERTA-* repos it will delete
python delete_certa_repos.py --yes    # deletes them permanently
```

## 2. (Optional) Rebuild the wrapper repos
Already built in `hf/lazem/`. Only needed if you change thresholds or the card template:
```bash
python build_hf_repos.py lazem
```

## 3. Upload, PRIVATE (flagship first, then the rest)
```bash
python stage_private.py --root hf --only NF4       # 11 flagship NF4 repos, private
python stage_private.py --root hf --only LLMINT8   # 9 deployable INT8 repos, private
python stage_private.py --root hf                  # everything remaining (RTN eval variants)
```
`stage_private.py` creates each repo **private** and uploads the folder. Re-running is safe (idempotent).

## 4. Group into one private collection (from the terminal)
```bash
python add_to_collection.py --namespace lazem      # creates the private collection + adds all 53 repos
```
This creates the collection "CERTA: Certified Compression for Clinical Decision Substitution" (private)
and adds the repos in order: 11 **NF4** flagship, then **LLMINT8**, then the RTN variants (models within
each tier follow the paper's Table 1 order). It is idempotent, so re-running just fills in anything
missing. It prints the collection URL at the end. Run it **after** step 3 (the repos must exist first).
Add `--public` only when you deliberately want the collection public. See `hf/COLLECTION.md` for the
long description text if you want to paste a richer write-up on the collection page.

## What each repo contains
```
certa_config.json        base model id, quantizer, bits, deployable flag, alpha options
certa_thresholds.json    exact calibrated per-class thresholds (alpha = 1/5/10%) from the study
certa_handler.py         load-time wrapper: downloads base, quantizes, certify-or-escalate at inference
README.md                model card (states: no weights re-hosted; GPU still needed to run)
LICENSE                  inherits the base model's license (VERIFY before any public release)
```

## Corrected base checkpoints (this release only; download_pairs.sh was for a different paper)
| Repo family        | Base model                                   |
|--------------------|----------------------------------------------|
| Llama-3-70B        | `meta-llama/Meta-Llama-3-70B-Instruct`       |
| Llama-3-8B         | `meta-llama/Meta-Llama-3-8B-Instruct`        |
| Mistral-7B         | `mistralai/Mistral-7B-Instruct-v0.1`         |
| MMed-Llama-3-8B    | `Henrychur/MMed-Llama-3-8B-EnIns`            |
| (others unchanged) | Qwen2.5-72B-Instruct, HuatuoGPT-o1-72B, Llama3-Med42-70B, Llama3-OpenBioLLM-70B, MMed-Llama3.1-70B, Llama-3.3-70B-Instruct, BioMistral-7B |

INT8 is intentionally skipped for `Llama-3-70B` and `Med42-70B` (LLM.int8() fails on those two
BF16-native 70B checkpoints), matching the paper. Total: 11 NF4 + 9 INT8 + 33 RTN = 53 repos.

## Weights pass (Falcon GPU) — build now, upload at preprint

Bake the deployable quantized weights (NF4 for all 11, LLM.int8() for the 9 that load) from the **exact
local eval checkpoints** on Falcon. RTN variants are bfloat16 (not memory-reducing) and are not baked.

**Now (GPU free) — Phase 1, build only, no upload.** Put this folder at
`/scratch/SCWF00175/shared/code/certa-hf-collection/` (or edit the two paths at the top of
`certa_quantize.slurm`) and make sure the env has `transformers`, `bitsandbytes`, `sentencepiece`:
```bash
sbatch certa_quantize.slurm      # writes safetensors into hf/lazem/CERTA-*-{NF4,LLMINT8}/
```
It reads each repo's local checkpoint from the map in the script (matches models.txt), skips repos whose
weights already exist (resumable), and pushes nothing. Storage: ~40 GB per 70B NF4, ~75 GB per 70B INT8.

**At preprint — Phase 2, upload (login node, internet).** Re-run the same private upload; it now carries
the weights that Phase 1 wrote:
```bash
python stage_private.py --root hf --only NF4
python stage_private.py --root hf --only LLMINT8
```
Verify each base model's license (especially the Llama and Qwen terms) before flipping any repo public.
To quantize a single repo by hand: `python quantize_and_push.py hf/lazem/CERTA-<model>-NF4 --base /scratch/SCWF00175/shared/models/<ckpt>`.
