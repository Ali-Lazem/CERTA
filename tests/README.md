# tests/

Reproducibility and robustness checks. Run from the repository root.

| File | Checks |
|------|--------|
| `cert_reproduce.py` | Recomputes the headline main-text numbers directly from `certa_core.py` (8-bit certified coverage ≥ 99.8%; the Qwen2.5-72B / HuatuoGPT-o1-72B ~quarter-BF16 4-bit frontier; escalation rates) and prints them for comparison against the manuscript. |
| `patient_level_check.py` | Stresses the patient-level exchangeability assumption behind the certificate: re-runs the frontier under a strict one-decision-per-patient restriction and reports the change, confirming the guarantee is not an artifact of the 16 patients that contribute a second decision. |
| `split_cert_check.py` | Validates the **exact** split-sample certificate (threshold grid on an independent selection subset, Clopper–Pearson bounds on a disjoint certification subset) and confirms it reproduces the reported full-calibration frontier within a median of 0.9 and at most 5.3 percentage points (Supplementary Table S7). |

```bash
python tests/cert_reproduce.py
python tests/patient_level_check.py
python tests/split_cert_check.py
```

These are lightweight (seconds to a couple of minutes) and need only the recorded outputs in `data/verify/` — no GPU.
