---
license: {{BASE_MODEL_LICENSE}}   # inherit the base model's license (e.g. llama3, apache-2.0)
base_model: {{BASE_MODEL_HF_ID}}
tags: [certa, certified-compression, clinical-nlp, selective-prediction, quantization]
---

# {{MODEL_NAME}} + CERTA (certified compression)

This repository adds a **CERTA certify-or-escalate layer** to a quantized clinical-verifier model.
CERTA decides, per decision, whether the quantized model may stand in for its full-precision
({{BASE_MODEL_HF_ID}}, bfloat16) reference at a stated disagreement-risk level, and otherwise
escalates the decision to the reference.

It does **not** re-host the base weights. At load time the handler pulls `{{BASE_MODEL_HF_ID}}`,
quantizes it (NF4 4-bit or INT8 8-bit) with bitsandbytes, and applies the calibrated thresholds in
`certa_thresholds.json`.

## What is and is not guaranteed
- **Certified:** compression-induced disagreement with the full-precision reference on the served
  decisions, bounded at the chosen risk level `alpha` with confidence `1 - delta` (delta = 0.10),
  for the **clinical accept/reject verification decision calibrated on PMC-Patients**.
- **NOT certified:** clinical correctness of the reference decision, and any guarantee on a **different
  task or population**. To deploy on new data, re-calibrate with `export_thresholds.py`.
- **Memory:** NF4 and INT8 reduce resident memory; round-to-nearest variants do not.
- **Escalation:** `escalate=True` decisions require the full-precision reference to be reachable; only
  zero-escalation operating points are standalone.

## Calibrated operating points (class-conditional, alpha = 5%)
| Quantizer | Escalation | Amortized size (% of full precision) | Certified? |
|-----------|-----------:|-------------------------------------:|:----------:|
| NF4 (4-bit) | {{NF4_ESC}}% | {{NF4_SIZE}}% | yes |
| INT8 (8-bit) | {{INT8_ESC}}% | {{INT8_SIZE}}% | {{INT8_CERT}} |

Full thresholds for alpha in {1%, 5%, 10%}, marginal and per-class, are in `certa_thresholds.json`.

## Usage
```python
from certa_handler import EndpointHandler          # CERTA_QUANT=nf4|int8, CERTA_ALPHA=0.01|0.05|0.10
h = EndpointHandler(".")
out = h({"inputs": "<your YES/NO verifier prompt>", "alpha": 0.05})
# {'decision': 'YES'|'NO', 'margin': ..., 'escalate': True|False, 'served_by': 'quantized'|'reference', ...}
```
If `escalate` is True, run the same prompt on the full-precision reference and use that decision.

## Citation
Lazem & Teahan, *Do We Need the Largest LLMs for Every Clinical Decision?* See the project repository
for code, recorded outputs and the full method.
