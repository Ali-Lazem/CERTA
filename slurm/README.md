# slurm/ — HPC job scripts

GPU-bound steps only. The certificate and the reproduction pipeline (`../src/`, `../tests/`) do **not**
need these; they are for regenerating quantized weights and validating that they load.

| File | Purpose |
|------|---------|
| `certa_bake.slurm` | Self-contained Phase-1 weight baking. Quantizes each deployable repository (NF4 4-bit and LLM.int8() 8-bit) from its exact local evaluation checkpoint, writes the weights into the repo folder, and pushes nothing. Preflight-checks dependencies and GPU; resumable (skips repos that already have weights). |
| `certa_quantize.slurm` | Earlier variant of the baking job (per-repo, via `../src/serve/quantize_and_push.py`). Kept for reference. |
| `certa_smoke.slurm` | Loads every baked deployable repository through the weights-aware handler and runs one certify-or-escalate decision, confirming the local quantized weights load without re-download. |

## Notes

- Edit the partition, account, GPU count and paths at the top of each script for your cluster.
- Environment must provide `transformers`, `bitsandbytes`, `sentencepiece`, `accelerate`, and CUDA; a single
  141 GB H200 holds even the INT8 72B model.
- Upload is deliberately separate from baking: run it from an internet-connected login node with
  `../src/serve/upload_public.py`, never from a compute node.
