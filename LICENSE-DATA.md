# Data licensing and provenance

This repository is designed to be fully reproducible while respecting the license of the source clinical corpus.

## Source corpus: PMC-Patients

The clinical decision contexts used in this study derive from **PMC-Patients** (Zhao et al., 2023), a public collection of 167,034 de-identified patient summaries extracted from case reports in PubMed Central. PMC-Patients is distributed under **CC BY-NC-SA 4.0**.

Because the source is CC BY-NC-SA 4.0, any redistributed text derived from it inherits that license. Accordingly:

- `data/verify/prompts.jsonl` includes, for each of the 2,500 decisions, the row identifier (`uid`, which maps to a PMC-Patients patient id), the task label (`ner` or `qar`), the verifier `system` and `user` prompt text (the patient context, the candidate and the verification question), and `nwords` (the `user` length in whitespace-separated tokens).
- The `system`/`user` prompt text derives from PMC-Patients and is therefore redistributed **under CC BY-NC-SA 4.0**: with attribution to PMC-Patients (Zhao et al., 2023), for **non-commercial** use only, under **share-alike** terms. Users must comply with these terms.

## What is redistributed here

- `data/verify/prompts.jsonl` — per-decision `uid`, task, and the verifier `system`/`user` input text plus `nwords`. The text is derived from PMC-Patients and is covered by **CC BY-NC-SA 4.0** (attribution, non-commercial, share-alike); the numeric fields are the authors' own.
- `data/verify/cert_*_{bnb,rtn}.npz` — numeric per-decision records: reference (BF16) decision, quantized decision, and decision margin for each model and quantizer family. These are numeric outputs of our own inference runs and are released for reproducibility.
- `data/verify/*.csv` — derived aggregates (external NER benchmark results, precomputed frontier and Mondrian tables).
- `data/verify/subgroup_*.txt` — subgroup breakdowns.

These numeric records fully reproduce every table, figure, and reported number in the paper (see the top-level `README.md`).

## Regenerating the prompts from the raw corpus

`data/verify/prompts.jsonl` already includes the verifier input text (redistributed under CC BY-NC-SA 4.0 as described above). To rebuild it from the raw corpus instead, download PMC-Patients from its official distribution and run `src/inference/extract_prompts.py`. See `src/inference/README.md`.

## Ethics

PMC-Patients consists of already-published, de-identified case-report summaries in the public domain of the biomedical literature. The study uses no private or identifiable patient data and involved no interaction with human subjects; it therefore did not require additional ethical approval or informed consent. Users are responsible for complying with the PMC-Patients license and their local regulations.

## Reference

Zhao, Z., et al. (2023). *PMC-Patients: A Large-scale Dataset of Patient Summaries and Relations for Benchmarking Retrieval-based Clinical Decision Support Systems.* (BibTeX key `zhao2023large` in `manuscript/certa.bib`.)
