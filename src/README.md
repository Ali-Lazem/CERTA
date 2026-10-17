# src/

The canonical analysis pipeline. **`certa_core.py` is the single source of truth**: every table, figure, and reported number is derived from it, so results cannot drift between artifacts.

Run all scripts **from the repository root** (paths resolve relative to the root):

```bash
python src/si_compute.py        # -> si_data.pkl
python src/si_compute2.py       # -> si_add.pkl
python src/si_compute3.py       # -> si_add3.pkl
python src/si_tables.py         # -> si_tables.tex   (Supplementary table source)
python src/build_figures.py     # -> fig2.pdf .. fig5.pdf
python src/build_si_figures.py  # -> figS1.pdf .. figS6.pdf
```

## Files

| File | Role |
|------|------|
| `certa_core.py` | Core certificate. Patient-grouped calibration/test split; Clopper–Pearson upper bound (`cp_upper`); per-decision margins (`arr`); threshold search (`_tau`); marginal frontier (`marg`, `marg_frontier`); class-conditional Mondrian frontier (`mond`, `mond_frontier`). Loads recorded outputs from `data/verify/`. |
| `si_compute.py` | Primary SI quantities → `si_data.pkl`: per-config coverage, decision distribution, marginal + Mondrian frontiers, split-sample sensitivity, task-wise coverage, escalation, prevalence stress test, external NER. |
| `si_compute2.py` | Patient-clustered bootstrap 95% CIs, class-specific calibration resolution, prevalence maxima, pooled certified/escalated margin ECDFs → `si_add.pkl`. |
| `si_compute3.py` | Escalation mechanism (resolution- vs risk-limited), threshold-grid sensitivity, repeated patient-disjoint split stability, class-conditional δ/α sensitivity → `si_add3.pkl`. |
| `si_tables.py` | Emits every Supplementary table macro to `si_tables.tex` from the three pickles. |
| `build_figures.py` | Main-text figures fig2–fig5 (all numbers recomputed from `certa_core`, none hardcoded). |
| `build_si_figures.py` | Supplementary figures figS1–figS6. |
| `inference/` | Prompt construction, GPU inference, and external NER benchmarking. See `inference/README.md`. Only needed to regenerate `data/verify/` from scratch. |

## Key certificate constants (in `certa_core.py`)

```
SEED = 0        # split seed
CALFRAC = 0.5   # patient-disjoint calibration fraction (rest is test)
SELFRAC = 0.20  # selection-subset fraction for the exact split-sample variant
GRID = 50       # threshold-grid size G (Bonferroni-corrected)
DELTA = 0.10    # certificate confidence budget (1 - delta)
A = 0.05        # target risk level alpha
```

Deployable quantizers searched for the frontier are NF4 (4-bit) and INT8 (`CONF_DEPLOY`); RTN 8/6/4 is a bf16-storage precision sweep retained only for the per-configuration table.

Setting selection = certification = full calibration recovers the reported higher-power certificate; the disjoint selection/certification split gives the exact finite-sample guarantee (Supplementary Table S7).
