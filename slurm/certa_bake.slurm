#!/bin/bash
#SBATCH --job-name=CERTA-bake
#SBATCH --output=certa_bake_%j.out
#SBATCH --account=SCWF00175_w_teahan_171
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=300G
#SBATCH --time=48:00:00

# =====================================================================================
# Self-contained CERTA weight-baking (Phase 1: NF4 + LLM.int8(), NO upload).
# Bakes each deployable repo from its EXACT local eval checkpoint. Nothing but the conda
# env is required (transformers, bitsandbytes, sentencepiece, accelerate, torch+CUDA).
# Resumable: repos that already have *.safetensors are skipped.
# =====================================================================================

module purge
module load Miniforge3/25.11.0-1
module load CUDA/12.1.1
source /scratch/SCWF00175/shared/envs/ai_pipeline_venv/bin/activate

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1            # load local checkpoints only; no network needed

# ---- folder that CONTAINS hf/lazem/CERTA-*  (confirmed on your system) ----
COLLECTION_DIR=/scratch/SCWF00175/shared/code/certa-hf-release-falcon
cd "$COLLECTION_DIR" || { echo "FATAL: cannot cd to $COLLECTION_DIR"; exit 1; }
ls -d hf/lazem/CERTA-*-NF4 >/dev/null 2>&1 || { echo "FATAL: no hf/lazem/CERTA-*-NF4 under $(pwd)"; exit 1; }
echo "working dir: $(pwd)"
echo "deployable repos: $(ls -d hf/lazem/CERTA-*-NF4 hf/lazem/CERTA-*-LLMINT8 2>/dev/null | wc -l) (expect 20)"

# ---- preflight: fail loudly if a dependency or GPU is missing ----
python - <<'PY' || { echo "FATAL: dependency/GPU preflight failed"; exit 3; }
import importlib, sys
for p in ["torch","transformers","bitsandbytes","sentencepiece","accelerate"]:
    try:
        m=importlib.import_module(p); print(p, getattr(m,"__version__","?"))
    except Exception as e:
        print("MISSING", p, "->", e); sys.exit(2)
import torch
print("CUDA available:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
sys.exit(0 if torch.cuda.is_available() else 4)
PY

# ---- bake ----
python - <<'PY'
import json, glob, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS = "/scratch/SCWF00175/shared/models"
BASE = {  # CERTA display name -> exact local eval checkpoint (from models.txt)
 "Qwen2.5-72B":"qwen2.5-72b-instruct", "HuatuoGPT-o1-72B":"huatuogpt-o1-72b",
 "Med42-70B":"med42-70b", "OpenBioLLM-70B":"openbiollm-70b",
 "Llama-3-70B":"llama3-70b-instruct-v3", "MMed-Llama-3.1-70B":"mmedllama3-70b",
 "Llama-3.3-70B":"llama3-70b", "BioMistral-7B":"biomistral-7b",
 "Llama-3-8B":"llama3-8b-instruct", "Mistral-7B":"mistral-7b-instruct-v0.1",
 "MMed-Llama-3-8B":"mmed-llama3-8b-enins"}

repos = sorted(glob.glob("hf/lazem/CERTA-*-NF4") + glob.glob("hf/lazem/CERTA-*-LLMINT8"))
print("repos to consider:", len(repos), flush=True)
done = fail = skip = 0
for d in repos:
    name = os.path.basename(d)
    if glob.glob(os.path.join(d, "*.safetensors")):
        print("SKIP (weights present):", name, flush=True); skip += 1; continue
    body, method = name.rsplit("-", 1)
    display = body[len("CERTA-"):]
    src = os.path.join(MODELS, BASE.get(display, ""))
    if not BASE.get(display) or not os.path.isdir(src):
        print("FAIL (no local checkpoint):", display, src, flush=True); fail += 1; continue
    bits = json.load(open(os.path.join(d, "certa_config.json")))["bits"]
    print(f"=== {name}: load {src} -> {bits}-bit ===", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(src)
        if bits == 4:
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            m = AutoModelForCausalLM.from_pretrained(src, quantization_config=qc, device_map="auto")
        else:  # 8-bit LLM.int8()
            m = AutoModelForCausalLM.from_pretrained(src, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
        m.save_pretrained(d, safe_serialization=True)
        tok.save_pretrained(d)
        print("wrote weights ->", d, flush=True); done += 1
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print("FAIL", name, "->", repr(e), flush=True); fail += 1
print(f"\nDONE: {done} baked, {skip} skipped, {fail} failed", flush=True)
PY

echo "==================================================="
echo " Phase 1 finished. Check the DONE line above for counts."
echo " Upload later from a LOGIN node with stage_private.py --root hf --only NF4 / --only LLMINT8"
echo "==================================================="
