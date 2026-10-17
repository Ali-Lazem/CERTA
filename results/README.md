# results/

Cached pipeline outputs — the committed versions of what `../src/si_compute*.py` produce. They let you rebuild tables and figures without recomputing, and serve as a reference to diff against after any change.

| File | Produced by | Contents |
|------|-------------|----------|
| `si_data.pkl` | `src/si_compute.py` | Per-configuration coverage, decision distribution, marginal + Mondrian frontiers (`mond`), split-sample sensitivity (`mond_split`), task-wise coverage, escalation, prevalence stress test, external NER. |
| `si_add.pkl` | `src/si_compute2.py` | Patient-clustered bootstrap 95% CIs, class-specific calibration resolution, prevalence maxima, certified/escalated margin ECDFs. |
| `si_add3.pkl` | `src/si_compute3.py` | Escalation mechanism per model (resolution- vs risk-limited) and min attainable CP, threshold-grid sensitivity, 20-split patient-disjoint stability, class-conditional δ/α sensitivity. |

## Verifying reproducibility

Re-running the pipeline from the repo root should regenerate these byte-for-byte (the split seed is fixed at 0):

```bash
python src/si_compute.py && python src/si_compute2.py && python src/si_compute3.py
# then compare the freshly written *.pkl in the working dir against results/
```

`src/si_tables.py` and both figure scripts consume these pickles (or recompute equivalently from `certa_core.py`).
