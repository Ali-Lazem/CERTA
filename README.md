# CERTA — Certify-or-Escalate for Clinical LLM Compression

**Do We Need the Largest LLMs for Every Clinical Decision?**

Ali H. Lazem¹² and William J. Teahan¹
¹ School of Computer Science and Electronic Engineering, Bangor University, UK
² Department of Computer Science, University of Thi-Qar, Iraq

This repository contains the code, recorded model outputs, statistical pipeline, figures, and manuscript sources for CERTA, a decision-level statistical certificate that determines when a quantized clinical language model may substitute for its full-precision (BF16) reference, and escalates the decisions it cannot certify back to that reference.

---

## What CERTA does

Model compression is usually treated as a single global precision choice for a whole model. CERTA treats it as a per-decision, risk-controlled policy. From the compressed model's own output on each input, CERTA computes a finite-sample statistical certificate (Learn-then-Test with a Clopper–Pearson upper bound and a Bonferroni correction over the threshold grid) that the compressed decision agrees with the BF16 reference. Decisions above the certified threshold are served by the small model; decisions below it are escalated to BF16. The result is an auditable "certified compression frontier": the smallest deployable footprint whose disagreement risk with BF16 is provably bounded at a chosen level.

Headline results (eleven models, three families, 7–72B; 2,500 clinical-verification decisions over 2,484 de-identified patients, patient-disjoint calibration and testing):

- 8-bit quantization certified at least **99.8%** of decisions for every model.
- Under a class-conditional (Mondrian) guarantee, a 4-bit scheme reduced the certified footprint to **about a quarter of BF16** for Qwen2.5-72B and HuatuoGPT-o1-72B, moving 70–72B models from a measured 142–146 GB across two 80 GB GPUs to 41–43 GB — within the capacity of a single 48 GB GPU.
- Independent disease named-entity benchmarking against human annotations showed the deployed configurations largely preserved F1.

---

## Repository layout

```
certa/
├── manuscript/     LaTeX sources: main text, Supplementary Information, tables, bibliography, cover letter
├── src/            The canonical analysis pipeline (single source of truth: certa_core.py)
│   └── inference/  Prompt construction, GPU inference, and external NER benchmarking (needs GPUs + weights)
├── data/           Recorded per-decision model outputs (.npz) and derived CSVs (data/verify/)
├── figures/        Publication PDFs: main figures (fig2–fig5) and supplementary (figS1–figS6)
├── results/        Cached pipeline outputs (si_data.pkl, si_add.pkl, si_add3.pkl)
└── tests/          Reproducibility and robustness checks
```

Each subfolder has its own `README.md` describing its contents.

---

## Reproducing the paper

Everything in the paper is reproducible **without GPUs or model weights** from the recorded decision outputs in `data/verify/`. Full re-inference from scratch (regenerating the `.npz`) requires the model weights and GPUs and is documented separately in `src/inference/README.md`.

```bash
# 1. environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. run everything from the repository root (paths are resolved relative to the root)
python src/si_compute.py        # -> si_data.pkl   (primary + split-sample sensitivity)
python src/si_compute2.py       # -> si_add.pkl    (bootstrap CIs, class resolution, prevalence)
python src/si_compute3.py       # -> si_add3.pkl   (escalation mechanism, grid/split stability)
python src/si_tables.py         # -> si_tables.tex (all Supplementary tables)
python src/build_figures.py     # -> fig2..fig5.pdf
python src/build_si_figures.py  # -> figS1..figS6.pdf

# 3. verification
python tests/cert_reproduce.py       # main-text numbers
python tests/patient_level_check.py  # patient-level exchangeability / one-decision-per-patient
python tests/split_cert_check.py     # exact split-sample certificate
```

A quick sanity check that data and code are wired correctly:

```bash
python -c "import sys; sys.path.insert(0,'src'); from certa_core import mond_frontier; \
print(mond_frontier('qwen2.5-72b')['size'])"   # -> 25.8
```

The committed `results/*.pkl`, `figures/*.pdf`, and `manuscript/si_tables.tex` are the exact artifacts used in the manuscript; re-running the pipeline should reproduce them.

---

## Method in one paragraph

Let `m_k(x)` be the compressed model's decision margin on input `x`, and let a disagreement with BF16 be the error event. On a patient-disjoint calibration set, CERTA searches a grid of thresholds `τ` and, for each, forms a Clopper–Pearson upper bound on the disagreement rate among decisions with `m_k(x) ≥ τ`, correcting for the grid by Bonferroni. The smallest certified footprint over deployable quantizers (NF4, INT8) at risk level α defines the frontier; a class-conditional variant certifies the accept and reject classes separately (Mondrian) so that neither class exceeds α. The reported certificate uses the full calibration set for maximum power; an exact finite-sample variant (split-sample: grid on an independent selection subset, bounds on a disjoint certification subset) reproduces the frontier within a median of 0.9 and at most 5.3 percentage points (Supplementary Table S7).

---

## Data and licensing

- **Code** (`src/`, `tests/`): MIT License (see `LICENSE`).
- **Recorded outputs** (`data/verify/*.npz`, CSVs): numeric decision records; released for reproducibility.
- **Patient text is not redistributed.** The clinical contexts derive from **PMC-Patients** (Zhao et al., 2023), which is licensed CC BY-NC-SA 4.0. `data/verify/prompts.jsonl` therefore contains only the row identifiers (`uid`) and task labels needed to reproduce the certificate from the recorded outputs. To regenerate the full-text prompts for re-inference, see `src/inference/README.md`. See `LICENSE-DATA.md` for details.

---

## Citation

If you use this code or the recorded outputs, please cite the paper (see `CITATION.cff`) and the PMC-Patients dataset:

> Zhao et al. *PMC-Patients: A Large-scale Dataset of Patient Summaries and Relations for Benchmarking Retrieval-based Clinical Decision Support Systems.* 2023.

---

## Contact

Ali H. Lazem — alazem@utq.edu.iq
