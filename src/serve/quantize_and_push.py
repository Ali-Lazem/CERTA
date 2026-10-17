#!/usr/bin/env python3
"""Produce the quantized weights for one generated CERTA repo directory and (optionally) upload it to
the Hugging Face Hub. RUN THIS ON A GPU MACHINE with the base model weights available and `huggingface_hub`
installed and logged in. It reads certa_config.json in the repo dir, quantizes the base model per the
recorded recipe, writes the checkpoint + tokenizer into the dir alongside the CERTA files, then uploads.

Usage:
    python quantize_and_push.py hf/lazem/CERTA-Qwen2.5-72B-NF4                          # build weights only
    python quantize_and_push.py hf/lazem/CERTA-Qwen2.5-72B-NF4 --base /path/to/ckpt     # from a local checkpoint
    python quantize_and_push.py hf/lazem/CERTA-Qwen2.5-72B-NF4 --push                   # build + upload to the Hub

Pass --base to quantize the EXACT local checkpoint used in the study (the Falcon eval directory), rather
than re-downloading from the Hub; the model card metadata (base_model id) is left unchanged.

Notes:
  * NF4/LLMINT8 use bitsandbytes and are saved as quantized checkpoints (memory-reducing).
  * RTN4/6/8 apply per-group round-to-nearest with an MSE clip search and are saved in bfloat16
    (a precision-evaluation variant, NOT memory-reducing) -- publish them clearly labelled as such.
  * Verify the base model's LICENSE and comply with its naming/use terms before --push.
"""
import argparse, json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def rtn_quantize_(model, bits, group=128, clips=(0.7, 0.8, 0.9, 1.0)):
    """In-place per-group symmetric round-to-nearest with an MSE clip search; bfloat16 storage."""
    for _, p in model.named_parameters():
        if p.dim() != 2:
            continue
        w = p.data.float()
        out, cols = w.shape
        g = group if cols % group == 0 else cols
        wr = w.reshape(out, -1, g)
        amax = wr.abs().amax(dim=-1, keepdim=True)
        best = None
        for c in clips:
            scale = (amax * c) / (2 ** (bits - 1) - 1)
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            q = torch.clamp(torch.round(wr / scale), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
            deq = q * scale
            err = (deq - wr).pow(2).mean()
            if best is None or err < best[0]:
                best = (err, deq)
        p.data.copy_(best[1].reshape(out, cols).to(p.dtype))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_dir")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--public", action="store_true", help="make the repo public (default: PRIVATE)")
    ap.add_argument("--base", default=None, help="local path to the exact base checkpoint to quantize "
                    "(overrides the HF id in certa_config.json; use the Falcon eval dir for exactness)")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.repo_dir, "certa_config.json")))
    q, bits = cfg["quantizer"], cfg["bits"]
    src = args.base or cfg["base_model"]     # what we load; card metadata base_model is left unchanged
    print(f"loading base from: {src}  ->  {q}/{bits}-bit  ->  {args.repo_dir}")
    tok = AutoTokenizer.from_pretrained(src)
    if q == "bnb" and bits == 4:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(src, quantization_config=bnb, device_map="auto")
    elif q == "bnb" and bits == 8:
        model = AutoModelForCausalLM.from_pretrained(src, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto")
    elif q == "rtn":
        model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.bfloat16, device_map="auto")
        rtn_quantize_(model, bits)
    else:
        raise ValueError(f"unknown quantizer {q}/{bits}")
    model.save_pretrained(args.repo_dir, safe_serialization=True)
    tok.save_pretrained(args.repo_dir)
    print(f"wrote quantized checkpoint + tokenizer to {args.repo_dir}")
    if args.push:
        from huggingface_hub import HfApi, create_repo
        create_repo(cfg["repo"], repo_type="model", private=not args.public, exist_ok=True)
        HfApi().upload_folder(folder_path=args.repo_dir, repo_id=cfg["repo"], repo_type="model")
        vis = "PUBLIC" if args.public else "PRIVATE"
        print(f"uploaded ({vis}) to https://huggingface.co/{cfg['repo']}  (verify LICENSE compliance first)")


if __name__ == "__main__":
    main()
