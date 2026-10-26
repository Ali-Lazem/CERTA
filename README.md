# CERTA — Certify-or-Escalate for Clinical LLM Compression

### Decision-level statistical certification for quantized clinical language models

[![arXiv](https://img.shields.io/badge/arXiv-2610.26937-b31b1b.svg)](https://arxiv.org/abs/2610.26937)
[![DOI](https://zenodo.org/badge/1342678081.svg)](https://doi.org/10.5281/zenodo.23435475)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-ffcc00.svg)](https://huggingface.co/collections/lazem/certa-certified-compression-for-clinical-decisions-6ad373efa6e143550bc15848)
[![Paper](https://img.shields.io/badge/paper-under%20review-lightgrey.svg)](#citation)

Ali H. Lazem¹² and William J. Teahan¹  
¹ School of Computer Science and Electronic Engineering, Bangor University, UK  
² Department of Computer Science, University of Thi-Qar, Iraq

CERTA is a decision-level statistical certificate that determines when a quantized clinical language model may substitute for its full-precision (BF16) reference, and escalates the decisions it cannot certify back to that reference. This repository holds the complete analysis pipeline, the recorded per-decision model outputs, the figure and table sources, the reproducibility tests, and the Hugging Face release tooling. The paper and Supplementary Information are hosted separately and linked below.

- **Paper (arXiv preprint):** https://arxiv.org/abs/2610.26937
- **Model collection (Hugging Face):** https://huggingface.co/collections/lazem/certa-certified-compression-for-clinical-decisions-6ad373efa6e143550bc15848
- **Archived snapshot (Zenodo):** https://doi.org/10.5281/zenodo.23435475

---

## What CERTA does

Model compression is usually treated as a single global precision choice for a whole model. CERTA treats it as a per-decision, risk-controlled policy. From the compressed model's own output on each input, CERTA computes a finite-sample statistical certificate (Learn-then-Test with a Clopper–Pearson upper bound and a Bonferroni correction over the threshold grid) that the compressed decision agrees with the BF16 reference. Decisions above the certified threshold are served by the small model; decisions below it are escalated to BF16. The result is an auditable "certified compression frontier": the smallest deployable footprint whose disagreement risk with BF16 is provably bounded at a chosen level.

Headline results (eleven models, three families, 7–72B; 2,500 clinical-verification decisions over 2,484 de-identified patients, patient-disjoint calibration and testing):

- 8-bit quantization certified at least **99.8%** of decisions for every model.
- Under a class-conditional (Mondrian) guarantee, a 4-bit scheme reduced the certified footprint to **about a quarter of BF16** for Qwen2.5-72B and HuatuoGPT-o1-72B, moving 70–72B models from a measured 142–146 GB across two 80 GB GPUs to 41–43 GB — within the capacity of a single 48 GB GPU.
- Independent disease named-entity benchmarking against human annotations showed the deployed configurations largely preserved task F1, confirming that reference agreement and downstream accuracy are complementary signals.

---

## Repository layout

```
certa/
├── src/            The canonical analysis pipeline (single source of truth: certa_core.py)
│   ├── inference/  Prompt construction, GPU inference, external NER benchmarking,
│   │               and the paired human-gold correctness analysis (needs GPUs + weights)
│   └── serve/      Hugging Face release: certificate packaging, license-compliant model cards, upload tooling
├── data/           Recorded per-decision model outputs (.npz) and derived CSVs (data/verify/)
├── figures/        Publication PDFs: main figures (fig2–fig5) and supplementary (figS1–figS6)
├── results/        Cached pipeline outputs (si_data.pkl, si_add.pkl, si_add3.pkl)
├── slurm/          HPC job scripts: weight quantization, smoke tests, and NER dump runs
└── tests/          Reproducibility and robustness checks
```

Each subfolder has its own `README.md` describing its contents. The LaTeX sources for the paper and Supplementary Information are maintained with the paper, not in this repository.

---

## Reproducing the paper

Everything in the paper is reproducible **without GPUs or model weights** from the recorded decision outputs in `data/verify/`. Full re-inference from scratch (regenerating the `.npz`) requires the model weights and GPUs and is documented separately in `src/inference/README.md`.

```bash
# 1. environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. regenerate every certificate-derived table and figure in one command:
bash reproduce_si.sh            # -> si_tables.tex, fig2-5.pdf, figS1-6.pdf (no model inference)

# ...or run the steps individually from the repository root (data/ and results/ resolved automatically):
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

The committed `results/*.pkl` and `figures/*.pdf` are the exact artifacts used in the paper; re-running the pipeline reproduces them. All figure values are recomputed from `certa_core.py` and the recorded outputs — none are hardcoded.

---

## Method in one paragraph

Let `m_k(x)` be the compressed model's decision margin on input `x`, and let a disagreement with BF16 be the error event. On a patient-disjoint calibration set, CERTA searches a grid of thresholds `τ` and, for each, forms a Clopper–Pearson upper bound on the disagreement rate among decisions with `m_k(x) ≥ τ`, correcting for the grid by Bonferroni. The smallest certified footprint over deployable quantizers (NF4, LLM.int8()) at risk level α defines the frontier; a class-conditional variant certifies the accept and reject classes separately (Mondrian) so that neither class exceeds α. The reported certificate uses the full calibration set for maximum power; an exact finite-sample variant (split-sample: grid on an independent selection subset, bounds on a disjoint certification subset) reproduces the frontier (see the Supplementary Information).

---

## External validation

`src/inference/bench_ner.py` benchmarks every model and configuration against human gold annotations on two disease-NER datasets (BC5CDR, NCBI-Disease); the recorded F1/precision/recall are in `data/verify/bench.csv` and drive Figure 5. `src/inference/ner_paired_analysis.py` adds the reviewer-requested **paired per-entity correctness analysis**: using per-document prediction dumps (`apply_dump_patch.py` adds a `--dump` flag to the benchmark; see `slurm/bench_all_dump.slurm`), it reports document-level paired bootstrap confidence intervals and the entity-level correctness transitions (preserved, newly-covered, lost, still-missed) between the BF16 reference and each deployed configuration.

---

## Model artifacts and release

The calibrated CERTA artifacts are released as a Hugging Face collection of **53 repositories** spanning all
eleven models and the full configuration set: NF4 4-bit and LLM.int8() 8-bit for deployment, and
round-to-nearest 8/6/4-bit as precision-evaluation variants. Each repository carries the CERTA layer
(per-class thresholds, machine-readable certificate, and a weights-aware certify-or-escalate handler) and
license-compliant packaging; base-model weights are not re-hosted, and released quantized checkpoints are
derivative works. LLM.int8() is intentionally absent for the two BF16-native 70B checkpoints
(Llama-3-70B, Med42-70B) where it collapses to zero certified coverage, as reported in the paper.

The release tooling lives in `src/serve/` (build the repositories, assemble the license kit, upload, and
group into the collection) and `slurm/` (GPU-side weight quantization). See `src/serve/README.md`.

---

## Data and licensing

- **Code** (`src/`, `tests/`): released under the license in `LICENSE`.
- **Released model artifacts** (Hugging Face collection): each repository ships the applicable upstream
  base-model license(s) for the quantized weights, plus an Apache-2.0 license (`LICENSE-CERTA`) covering the
  CERTA-authored files (handler, thresholds, config, card text). HuatuoGPT-o1-72B is packaged conservatively
  under both Apache-2.0 and the Qwen license (its Qwen2.5-72B base).
- **Recorded outputs** (`data/verify/*.npz`, CSVs): numeric decision records; released for reproducibility.
- **Patient text is not redistributed.** The clinical contexts derive from **PMC-Patients** (Zhao et al., 2023), licensed CC BY-NC-SA 4.0. `data/verify/prompts.jsonl` therefore contains only the row identifiers (`uid`) and task labels needed to reproduce the certificate from the recorded outputs. To regenerate the full-text prompts for re-inference, see `src/inference/README.md`. See `LICENSE-DATA.md` for details.

---

## Citation
## Citation

```bibtex
@article{lazem2026we,
  title         = {Do We Need the Largest LLMs for Every Clinical Decision?},
  author        = {Lazem, Ali H. and Teahan, William J.},
  journal       = {arXiv preprint arXiv:2610.26937},
  year          = {2026},
  eprint        = {2610.26937},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.
If you use this code or the recorded outputs, please cite the paper (see `CITATION.cff`) and the PMC-Patients dataset:

> Zhao et al. *PMC-Patients: A Large-scale Dataset of Patient Summaries and Relations for Benchmarking Retrieval-based Clinical Decision Support Systems.* 2023.

---

## Contact

Ali H. Lazem — lhl23prg@bangor.ac.uk
