#!/bin/bash
#SBATCH --job-name=bench_ner_dump
#SBATCH --output=bench_ner_dump_%j.out
#SBATCH --account=SCWF00175_w_teahan_171
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=72:00:00

module purge
module load Miniforge3/25.11.0-1
module load CUDA/12.1.1
source /scratch/SCWF00175/shared/envs/ai_pipeline_venv/bin/activate
cd /scratch/SCWF00175/shared/code
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# external NER benchmarks (gold labels), both confirmed parsing (BC5CDR jsonl, NCBI inline)
DATASETS="/scratch/SCWF00175/shared/datasets/bc5cdr_disease /scratch/SCWF00175/shared/datasets/ncbi_disease"

# IMPORTANT: a FRESH csv (NOT bench.csv) so no config is skipped -> every config recomputes AND dumps.
# bench_dump.csv must come out identical to bench.csv (that is the reproduce-before-trust check).
CSV=bench_dump.csv
DUMP=preds.jsonl                 # per-document gold/pred dump for ner_paired_analysis.py
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# RTN: full (=BF16) + 8/6/4-bit, all models in models.txt, both datasets.
while read -r label path _rest; do
    [[ -z "$label" || "$label" == \#* ]] && continue
    [[ ! -d "$path" ]] && { echo "SKIP $label"; continue; }
    for DS in $DATASETS; do
        echo ""; echo ">>> $label on $(basename "$DS")"
        python bench_ner_dump.py --model "$path" --label "$label" --data "$DS" \
            --bits 8,6,4 --quantizer rtn --max-docs 200 --max-new-tokens 128 --match overlap --chat \
            --csv "$CSV" --dump "$DUMP" \
            || echo "FAILED $label on $DS, continuing"
    done
done < models.txt

echo ""; echo "=== BENCHMARK TABLE ($CSV) — compare to bench.csv ==="
column -t -s, "$CSV" 2>/dev/null || cat "$CSV"
echo "=== dumped lines so far: $(wc -l < "$DUMP" 2>/dev/null) ==="
echo "=== DONE (now submit bench_nf4_dump.slurm) ==="
