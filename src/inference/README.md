# src/inference/

Scripts that generate the raw material in `data/verify/` from scratch. **These require model weights and GPUs** (a 70B BF16 reference needs at least two 80 GB accelerators) and are *not* needed to reproduce the paper — the recorded outputs are already provided. Use these only to rebuild from the models.

| File | Role | Output |
|------|------|--------|
| `extract_prompts.py` | Builds verifier prompts (accept/reject decisions) from clinical-verifier candidate records using the verbatim templates in the SI. Auto-detects task (NER / QA-relevance). | `prompts.jsonl` (with full context text) |
| `forward_prop_gpu.py` | Runs each model at BF16 and at each quantization (RTN 8/6/4 in place; bitsandbytes NF4/INT8), records the reference decision, the quantized decision, and the decision margin per input, and measures throughput and peak VRAM. | `cert_<model>_<family>.npz` |
| `bench_ner.py` | Independent external benchmark: entity-level precision/recall/F1 of full vs quantized models against gold disease spans (NCBI-Disease, BC5CDR-disease). | rows appended to `bench.csv` |

## Pipeline order

```bash
# 1. Download PMC-Patients yourself (CC BY-NC-SA 4.0) and prepare candidate records.
# 2. Build prompts:
python extract_prompts.py --in verifier_candidates.jsonl --task ner --out prompts.jsonl

# 3. Record decisions + margins for each model (example: one 70B):
python forward_prop_gpu.py --model /path/to/Qwen2.5-72B --prompts prompts.jsonl \
    --answers "YES,NO" --quantizer bnb --bits 8,4    # -> cert_qwen2.5-72b_bnb.npz
python forward_prop_gpu.py --model /path/to/Qwen2.5-72B --prompts prompts.jsonl \
    --answers "YES,NO" --quantizer rtn --bits 8,6,4  # -> cert_qwen2.5-72b_rtn.npz

# 4. External NER benchmark:
python bench_ner.py --model /path/to/model --label qwen2.5-72b \
    --data /path/to/datasets/bc5cdr_disease --bits 8,6,4 --quantizer rtn --csv bench.csv
```

## Models evaluated (11; 3 families; 7–72B)

Qwen2.5-72B, HuatuoGPT-o1-72B, Med42-70B, OpenBioLLM-70B, MMed-Llama-3.1-70B, Llama-3.3-70B, Llama-3-70B, Llama-3-8B, MMed-Llama-3-8B, Mistral-7B, BioMistral-7B. Exact Hugging Face checkpoints are listed in the model inventory table of the Supplementary Information.

## Quantizers

- **RTN 8/6/4** — per-group (128) round-to-nearest with MSE clipping, stored in bf16. A precision sweep; it does **not** reduce memory, so it is a baseline, not a deployment option.
- **bitsandbytes NF4 (4-bit)** and **INT8 (LLM.int8())** — the memory-reducing, deployable configurations. INT8 fails to load on the two BF16-native 70B models.

## Dependencies

Not pinned in `requirements.txt`. Typically: `torch`, `transformers`, `vllm`, `accelerate`, `bitsandbytes`, `datasets`, matched to your CUDA version.
