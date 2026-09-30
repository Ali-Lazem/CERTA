# CERTA: Certified Compression for Clinical Decision Substitution

A Hugging Face collection of quantized clinical-verifier models wrapped with the CERTA certify-or-escalate
certificate. Group these repositories into one collection named above under the **AliLazem** namespace.

Naming: `CERTA-<model>-<METHOD>`, where METHOD is the actual compression scheme (NF4, LLMINT8, RTN4/6/8),
so the method is explicit rather than a bare "4-bit"/"8-bit" label.

- **Deployable (memory-reducing): 20 repos** -- the NF4 releases (principal 4-bit result) and the
  applicable LLM.int8() releases. Feature these.
- **Evaluation variants: 33 repos** -- RTN4/6/8, which retain bfloat16 storage and are published for
  reproducibility of the paper's precision sweep, not as deployments.
- **INT8 skipped** on models where LLM.int8() fails to load (0% certified coverage), as reported in the paper.

Each repository is a quantized derivative of its named base model plus the CERTA layer; it does not
fine-tune the base model. The certificate is conditional on the clinical accept/reject task, the
PMC-Patients calibration distribution, alpha, the thresholding procedure and the reference-model definition.

Total generated: 53 repositories. See MANIFEST.json.
