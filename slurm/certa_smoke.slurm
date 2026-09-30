#!/bin/bash
#SBATCH --job-name=CERTA-smoke
#SBATCH --output=certa_smoke_%j.out
#SBATCH --account=SCWF00175_w_teahan_171
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=03:00:00

# =====================================================================================
# Smoke-test every baked deployable repo (11 NF4 + 9 LLM.int8()): confirm the weights-aware
# handler loads the LOCAL quantized weights (no re-download) and returns a well-formed
# certify-or-escalate decision. One H200 (141 GB) holds even the INT8 72B checkpoint.
# =====================================================================================

module purge
module load Miniforge3/25.11.0-1
module load CUDA/12.1.1
source /scratch/SCWF00175/shared/envs/ai_pipeline_venv/bin/activate

COLLECTION_DIR=/scratch/SCWF00175/shared/code/certa-hf-release-falcon
cd "$COLLECTION_DIR" || { echo "FATAL: cannot cd to $COLLECTION_DIR"; exit 1; }

export HF_HUB_OFFLINE=1          # load local baked weights only; never re-download a base
export TOKENIZERS_PARALLELISM=false

echo "node: $(hostname)  GPUs: $CUDA_VISIBLE_DEVICES"
python - <<'PY'
import glob, os, time, gc, torch
from certa_handler import EndpointHandler

repos = sorted(glob.glob("hf/lazem/CERTA-*-NF4") + glob.glob("hf/lazem/CERTA-*-LLMINT8"))
prompt = {"inputs": "Question: Does the note support the diagnosis? Answer YES or NO.\nAnswer:", "alpha": 0.05}
print(f"testing {len(repos)} deployable repos\n")
ok = bad = 0
for d in repos:
    name = os.path.basename(d)
    try:
        t = time.time()
        h = EndpointHandler(d)
        out = h(prompt)
        assert out["decision"] in ("YES", "NO") and isinstance(out["escalate"], bool)
        print(f"PASS {name:38s} {time.time()-t:6.1f}s  served_by={out['served_by']:9s} margin={out['margin']}", flush=True)
        ok += 1
        del h; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        print(f"FAIL {name:38s} {repr(e)[:140]}", flush=True); bad += 1
        gc.collect(); torch.cuda.empty_cache()
print(f"\nSMOKE DONE: {ok} passed, {bad} failed of {len(repos)}")
PY
echo "==================================================="
echo " Smoke test finished. Look for 'SMOKE DONE: 20 passed, 0 failed'."
echo "==================================================="
