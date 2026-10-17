#!/usr/bin/env python3
"""CERTA certify-or-escalate inference handler (weights-aware).

Load behaviour:
  * If the repository contains baked quantized weights (`*.safetensors`), load them **directly** from the
    repo folder. A model saved after bitsandbytes NF4/INT8 quantization carries its `quantization_config`
    in `config.json`, so `from_pretrained(path)` reconstructs the quantized model without re-downloading or
    re-quantizing the base.
  * Otherwise fall back to the wrapper form: download the base model named in `certa_config.json` and
    quantize it at load time.

For each accept/reject decision it serves the quantized decision only when its margin clears the CERTA
threshold for its predicted class at the requested risk level alpha; otherwise it returns escalate=True
(refer the decision to the full-precision reference). Requires a CUDA GPU + bitsandbytes for NF4/INT8.

Scope: the thresholds certify the clinical accept/reject verification decision on the PMC-Patients
calibration distribution. They carry NO finite-sample guarantee on a different task or population;
re-calibrate for a new distribution. RTN variants (bfloat16 storage) are not runnable through this handler.
"""
import json, os, glob, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class EndpointHandler:
    def __init__(self, path="."):
        self.cfg = json.load(open(os.path.join(path, "certa_config.json")))
        self.th = json.load(open(os.path.join(path, "certa_thresholds.json")))
        self.alpha = str(self.cfg.get("default_alpha", 0.05))
        base, q, bits = self.cfg["base_model"], self.cfg["quantizer"], self.cfg["bits"]
        has_local_weights = bool(glob.glob(os.path.join(path, "*.safetensors")))

        if has_local_weights:
            # Baked quantized weights present: load them directly from the repo (no re-download/re-quantize).
            self.model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
            self.tok = AutoTokenizer.from_pretrained(path)
        elif q == "bnb" and bits == 4:
            qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(base, quantization_config=qc, device_map="auto")
            self.tok = AutoTokenizer.from_pretrained(base)
        elif q == "bnb" and bits == 8:
            self.model = AutoModelForCausalLM.from_pretrained(
                base, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
            self.tok = AutoTokenizer.from_pretrained(base)
        else:
            raise NotImplementedError(
                "This is an RTN evaluation variant (bfloat16 storage, not memory-reducing); "
                "it is provided for reproducibility and is not runnable through this handler.")

        self.model.eval()
        yes, no = self.th["answer_tokens"]["YES"], self.th["answer_tokens"]["NO"]
        self.yid = self.tok.encode(yes, add_special_tokens=False)[-1]
        self.nid = self.tok.encode(no, add_special_tokens=False)[-1]

    @torch.no_grad()
    def __call__(self, data):
        prompt = data["inputs"] if isinstance(data, dict) else data
        alpha = str(data.get("alpha", self.alpha)) if isinstance(data, dict) else self.alpha
        th = self.th["alpha"][alpha]["class_conditional"]
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        logits = self.model(**ids).logits[0, -1]
        zy, zn = float(logits[self.yid]), float(logits[self.nid])
        decision = "YES" if zy >= zn else "NO"
        margin = abs(zy - zn)
        tau = th["tau_yes"] if decision == "YES" else th["tau_no"]
        escalate = (tau is None) or (margin < tau)
        return {"decision": decision, "margin": round(margin, 4), "threshold": tau, "alpha": float(alpha),
                "escalate": escalate, "served_by": "reference" if escalate else "quantized",
                "note": "escalate=True: refer this decision to the full-precision reference model"}
