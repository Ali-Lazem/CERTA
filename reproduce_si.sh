#!/usr/bin/env bash
# =============================================================================
# CERTA -- regenerate every certificate-derived Supplementary table and figure
# from the recorded per-decision outputs in data/verify/ (no model inference).
#
# Run from the repository root:   bash reproduce_si.sh
# Outputs (si_data.pkl, si_add.pkl, si_add3.pkl, si_tables.tex, fig*.pdf,
# figS*.pdf) are written to the current working directory.
#
# Dependencies: see environment.yml / requirements.txt (NumPy, SciPy, matplotlib).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/6] certificates + NER summaries  -> si_data.pkl"
python src/si_compute.py
echo "[2/6] bootstrap CIs, class resolution, prevalence  -> si_add.pkl"
python src/si_compute2.py
echo "[3/6] escalation mechanism, grid/split stability  -> si_add3.pkl"
python src/si_compute3.py
echo "[4/6] Supplementary tables  -> si_tables.tex"
python src/si_tables.py
echo "[5/6] main-text figures  -> fig2.pdf .. fig5.pdf"
python src/build_figures.py
echo "[6/6] supplementary figures  -> figS1.pdf .. figS6.pdf"
python src/build_si_figures.py

echo
echo "Done. The external disease-NER F1 table and Figure 5 are regenerated from"
echo "the recorded benchmark outputs in data/verify/bench.csv; regenerating those"
echo "benchmark outputs, and the per-entity paired analysis, requires re-running"
echo "inference on the public BC5CDR-Disease / NCBI-Disease corpora (see src/inference/)."
