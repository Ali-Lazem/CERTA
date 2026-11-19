# data/

Recorded per-decision model outputs and derived aggregates. Everything here is numeric — no patient text is redistributed (see `../LICENSE-DATA.md`). These files fully reproduce every number, table, and figure in the paper.

All files live in `verify/`.

## Per-decision records — `cert_<model>_<family>.npz`

One file per model per quantizer family (11 models × {`bnb`, `rtn`} = 22 files). Each holds aligned arrays of length **2,500** (one entry per decision), indexed identically to `prompts.jsonl`.

`cert_<model>_bnb.npz` (bitsandbytes: INT8 and NF4):

| key | meaning |
|-----|---------|
| `ref_d` | BF16 reference decision (0/1) |
| `d8`, `d4` | quantized decision at INT8 and NF4 |
| `m8`, `m4` | decision margin at INT8 and NF4 |

`cert_<model>_rtn.npz` (round-to-nearest 8/6/4): keys `ref_d`, `d8`, `d6`, `d4`, `m8`, `m6`, `m4`.

A disagreement (the error event the certificate bounds) is `d != ref_d`. The margin is the compressed model's confidence signal used as the certificate score.

## `prompts.jsonl`

2,500 lines, one JSON object per decision: `{"uid": ..., "task": "ner"|"qar"}`. `uid` maps to a PMC-Patients patient id (`<patient>-<n>`); patients with more than one decision share a `uid` prefix, which is what the patient-disjoint split and patient-clustered bootstrap key on. **The prompt text is intentionally omitted** — see `../LICENSE-DATA.md` to regenerate it.

## Derived CSVs

| file | columns |
|------|---------|
| `bench.csv` | `model, dataset, quantizer, config, precision, recall, f1` — external NER (NCBI-Disease, BC5CDR-disease) vs human labels |
| `frontier_{bnb,rtn}.csv` | `model, family, alpha, bit, certCov, validity, escalation, amortBits, sizePct` — marginal certified frontier |
| `mondrian_{bnb,rtn}.csv` | `model, alpha, bit, cov, covYES, covNO, amortBits, sizePct` — class-conditional (Mondrian) frontier |
| `subgroup_{nf4,rtn8}_a05.txt` | subgroup coverage/disagreement breakdowns at α = 5% |

The CSVs are convenience aggregates; the pipeline recomputes the same quantities directly from the `.npz` via `certa_core.py`, so results never depend on the CSVs being in sync.

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
