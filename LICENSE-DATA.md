# Data licensing and provenance

This repository is designed to be fully reproducible while respecting the license of the source clinical corpus.

## Source corpus: PMC-Patients

The clinical decision contexts used in this study derive from **PMC-Patients** (Zhao et al., 2023), a public collection of 167,034 de-identified patient summaries extracted from case reports in PubMed Central. PMC-Patients is distributed under **CC BY-NC-SA 4.0**.

Because the source is CC BY-NC-SA:

- We **do not redistribute** the patient-context prompt text in this repository.
- `data/verify/prompts.jsonl` contains only the fields required to reproduce the statistical certificate from the recorded model outputs: the row identifier (`uid`, which maps to a PMC-Patients patient id) and the task label (`ner` or `qar`). It does **not** contain the `system` or `user` prompt text.

## What is redistributed here

- `data/verify/cert_*_{bnb,rtn}.npz` — numeric per-decision records: reference (BF16) decision, quantized decision, and decision margin for each model and quantizer family. These are numeric outputs of our own inference runs and are released for reproducibility.
- `data/verify/*.csv` — derived aggregates (external NER benchmark results, precomputed frontier and Mondrian tables).
- `data/verify/subgroup_*.txt` — subgroup breakdowns.

These numeric records fully reproduce every table, figure, and reported number in the paper (see the top-level `README.md`).

## Regenerating the full-text prompts

To rebuild the full `prompts.jsonl` (with patient-context text) — required only if you want to re-run model inference from scratch — download PMC-Patients yourself from its official distribution and run `src/inference/extract_prompts.py`. See `src/inference/README.md`.

## Ethics

PMC-Patients consists of already-published, de-identified case-report summaries in the public domain of the biomedical literature. The study uses no private or identifiable patient data and involved no interaction with human subjects; it therefore did not require additional ethical approval or informed consent. Users are responsible for complying with the PMC-Patients license and their local regulations.

## Reference

Zhao, Z., et al. (2023). *PMC-Patients: A Large-scale Dataset of Patient Summaries and Relations for Benchmarking Retrieval-based Clinical Decision Support Systems.* (BibTeX key `zhao2023large` in `manuscript/certa.bib`.)
