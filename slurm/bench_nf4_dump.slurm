#!/bin/bash
#SBATCH --job-name=bench_nf4_dump
#SBATCH --output=bench_nf4_dump_%j.out
#SBATCH --account=SCWF00175_w_teahan_171
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=200G
#SBATCH --time=12:00:00

# NF4 4-bit external disease-NER, appending to the SAME fresh bench_dump.csv and preds.jsonl as the
# RTN run. Submit this AFTER bench_all_dump.slurm has finished, so "full" (=BF16) is already in
# bench_dump.csv and is skipped here (dumped once, by the RTN run). Resume-safe: greedy decoding is
# deterministic, so any re-dumped (model,config,doc) line is identical and harmlessly deduped.

module purge
module load Miniforge3/25.11.0-1
module load CUDA/12.1.1
source /scratch/SCWF00175/shared/envs/ai_pipeline_venv/bin/activate
cd /scratch/SCWF00175/shared/code
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

DATASETS="/scratch/SCWF00175/shared/datasets/bc5cdr_disease /scratch/SCWF00175/shared/datasets/ncbi_disease"
CSV=bench_dump.csv
DUMP=preds.jsonl
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

while read -r label path _rest; do
    [[ -z "$label" || "$label" == \#* ]] && continue
    [[ ! -d "$path" ]] && { echo "SKIP $label (no path)"; continue; }
    for DS in $DATASETS; do
        echo ""; echo ">>> $label [NF4] on $(basename "$DS")"
        python bench_ner_dump.py --model "$path" --label "$label" --data "$DS" \
            --bits 4 --quantizer bnb --max-docs 200 --max-new-tokens 128 --match overlap --chat \
            --csv "$CSV" --dump "$DUMP" \
            || echo "FAILED $label [NF4] on $DS, continuing"
    done
done < models.txt

echo ""; echo "=== NF4 (bnb,4) rows now in $CSV ==="
head -1 "$CSV"; grep ",bnb,4," "$CSV" 2>/dev/null | column -t -s, || true
echo "=== total dumped lines: $(wc -l < "$DUMP" 2>/dev/null) ==="
echo "=== DONE (now run ner_paired_analysis.py) ==="
