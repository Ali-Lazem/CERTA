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
