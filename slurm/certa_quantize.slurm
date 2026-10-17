#!/bin/bash
#SBATCH --job-name=CERTA-quant
#SBATCH --output=certa_quant_%j.out
#SBATCH --account=SCWF00175_w_teahan_171
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=300G
#SBATCH --time=48:00:00

# =====================================================================================
# CERTA weight-baking, PHASE 1 (build weights, NO upload).
#
# Bakes the DEPLOYABLE quantized weights only -- NF4 (11 models) and LLM.int8() (9 models;
# the 2 BF16-native 70B checkpoints where LLM.int8() fails have no INT8 repo). RTN variants are
# bfloat16 storage (not memory-reducing) and are intentionally NOT baked; they stay metadata-only.
#
# Each model is quantized from its EXACT local eval checkpoint (the dir behind the paper's numbers),
# not a fresh Hub download, via quantize_and_push.py --base. Weights are written into the repo folder.
# The compute node is offline (HF_HUB_OFFLINE=1); loading local dirs needs no network. Resumable:
# repos whose weights already exist are skipped.
#
# After this job: upload from a LOGIN node (internet) at preprint time, e.g.
#     cd /scratch/SCWF00175/shared/code/certa-hf-collection
#     python stage_private.py --root hf --only NF4
#     python stage_private.py --root hf --only LLMINT8
# =====================================================================================

module purge
module load Miniforge3/25.11.0-1
module load CUDA/12.1.1
source /scratch/SCWF00175/shared/envs/ai_pipeline_venv/bin/activate   # must have: transformers, bitsandbytes, sentencepiece

# ---- Point this at the folder that CONTAINS hf/lazem/CERTA-* (edit if yours differs) ----
COLLECTION_DIR=/scratch/SCWF00175/shared/code/certa-hf-collection
cd "$COLLECTION_DIR" || { echo "FATAL: cannot cd to $COLLECTION_DIR"; exit 1; }
if ! ls -d hf/lazem/CERTA-*-NF4 >/dev/null 2>&1; then
  echo "FATAL: no hf/lazem/CERTA-*-NF4 under $(pwd)."
  echo "       Set COLLECTION_DIR to the folder containing hf/lazem/  (find it with:"
  echo "       find /scratch/SCWF00175 -maxdepth 4 -type d -name 'CERTA-*-NF4')"
  exit 1
fi
echo "working dir: $(pwd)"
echo "deployable repos found: $(ls -d hf/lazem/CERTA-*-NF4 hf/lazem/CERTA-*-LLMINT8 2>/dev/null | wc -l) (expect 20)"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1          # load local checkpoints only; no network on the compute node

SCRIPT="$COLLECTION_DIR/quantize_and_push.py"   # shipped inside this folder
MODELS=/scratch/SCWF00175/shared/models

# CERTA display name -> exact local eval checkpoint (from models.txt)
declare -A BASE=(
  [Qwen2.5-72B]=$MODELS/qwen2.5-72b-instruct
  [HuatuoGPT-o1-72B]=$MODELS/huatuogpt-o1-72b
  [Med42-70B]=$MODELS/med42-70b
  [OpenBioLLM-70B]=$MODELS/openbiollm-70b
  [Llama-3-70B]=$MODELS/llama3-70b-instruct-v3
  [MMed-Llama-3.1-70B]=$MODELS/mmedllama3-70b
  [Llama-3.3-70B]=$MODELS/llama3-70b
  [BioMistral-7B]=$MODELS/biomistral-7b
  [Llama-3-8B]=$MODELS/llama3-8b-instruct
  [Mistral-7B]=$MODELS/mistral-7b-instruct-v0.1
  [MMed-Llama-3-8B]=$MODELS/mmed-llama3-8b-enins
)

echo "==================================================="
echo " CERTA weight-baking (Phase 1: NF4 + LLM.int8(), no upload)"
echo " node: $(hostname)   GPUs: $CUDA_VISIBLE_DEVICES"
echo "==================================================="

for group in NF4 LLMINT8; do
  for d in hf/lazem/CERTA-*-${group}; do
    [ -d "$d" ] || continue
    if ls "$d"/*.safetensors >/dev/null 2>&1; then
      echo "SKIP (weights already present): $d"; continue
    fi
    name=$(basename "$d"); tmp=${name#CERTA-}; display=${tmp%-${group}}
    base=${BASE[$display]}
    if [ -z "$base" ] || [ ! -d "$base" ]; then
      echo "FAILED (no local checkpoint for '$display'): $d"; continue
    fi
    echo "=== quantizing $d  from $base  ($(date '+%H:%M:%S')) ==="
    python "$SCRIPT" "$d" --base "$base" || echo "FAILED: $d"
  done
done

echo "==================================================="
echo " Phase 1 complete. Quantized weights written into hf/lazem/CERTA-*-{NF4,LLMINT8}/ ."
echo " Upload later from a LOGIN node (internet) with stage_private.py (see header)."
echo "==================================================="
