#!/usr/bin/env python3
"""
forward_prop_gpu.py — certified decision-preserving compression on a real 70B,
swept over bit-widths, with the full rigour set:

  * quantizer:  RTN (per-group, in place) OR bitsandbytes NF4/INT8 (real quantizer)
  * throughput + peak VRAM measured per precision   (--> the hardware claim)
  * Learn-then-Test conformal certificate (Bonferroni over a fixed threshold grid)
  * bootstrap 95% CIs on coverage / validity / router-overall
  * reasoning-model diagnostic: fraction of cases whose top token is a decision token
    (flags models like HuatuoGPT-o1 that reason before answering)

Reads JSONL prompts with system/user fields (from make_prompts.py / extract_prompts.py)
or a plain .txt (one prompt per line). Certifies the argmax over --answers tokens
against the full-precision model.

------------------------------------------------------------------------------
HOW TO RUN  (GPU allocation; 70B bf16 reference needs >=2 H200)
    python forward_prop_gpu.py --model /path/to/model \
        --prompts prompts.jsonl --answers "YES,NO" \
        --quantizer rtn --bits 8,6,4 --group 128 --alpha 0.05 --chat \
        --label mmed-3.1-70b --csv cert_all.csv
    # real-quantizer arm (VRAM actually drops): --quantizer bnb --bits 8,4
------------------------------------------------------------------------------
"""
import argparse, json, math, gc, sys, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--answers", default="YES,NO")
ap.add_argument("--bits", default="8,6,4")
ap.add_argument("--quantizer", choices=["rtn", "bnb"], default="rtn")
ap.add_argument("--group", type=int, default=128)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--delta", type=float, default=0.10)
ap.add_argument("--cal-frac", type=float, default=0.5, help="calibration fraction (rest=test); raise it to certify small alpha")
ap.add_argument("--grid", type=int, default=50, help="threshold grid size for Learn-then-Test")
ap.add_argument("--n-boot", type=int, default=1000)
ap.add_argument("--max-prompts", type=int, default=0)
ap.add_argument("--chat", action="store_true")
ap.add_argument("--label", default="model")
ap.add_argument("--csv", default="")
ap.add_argument("--out", default="cert_real.npz")
A = ap.parse_args()
bits_list = [int(b) for b in A.bits.split(",")]
if A.quantizer == "bnb":
    bits_list = [b for b in bits_list if b in (4, 8)]   # bnb supports 4/8 only
    if not bits_list: sys.exit("bnb supports only --bits 4 and/or 8")

def load_prompts(path):
    P = []
    for line in open(path):
        line = line.rstrip("\n")
        if not line.strip(): continue
        P.append(json.loads(line) if line.lstrip().startswith("{") else {"user": line})
    return P
prompts = load_prompts(A.prompts)
if A.max_prompts: prompts = prompts[:A.max_prompts]

tok = AutoTokenizer.from_pretrained(A.model)
answers = A.answers.split(",")
ans_ids = [tok.encode(a, add_special_tokens=False)[0] for a in answers]
print(f"{len(prompts)} prompts | answers {dict(zip(answers, ans_ids))} | "
      f"quantizer={A.quantizer} sweep {bits_list}-bit", flush=True)

def load_model(cfg):
    if cfg == "full":
        return AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16, device_map="auto").eval()
    if A.quantizer == "bnb":
        from transformers import BitsAndBytesConfig
        qc = (BitsAndBytesConfig(load_in_8bit=True) if cfg == 8 else
              BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True))
        return AutoModelForCausalLM.from_pretrained(A.model, quantization_config=qc, device_map="auto").eval()
    m = AutoModelForCausalLM.from_pretrained(A.model, dtype=torch.bfloat16, device_map="auto").eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 2: p.data.copy_(qgroup_(p.data.float(), cfg, A.group).to(p.dtype))
    return m

@torch.no_grad()
def qgroup_(W, k, G):
    out, inn = W.shape; pad = (-inn) % G
    X = torch.cat([W, torch.zeros(out, pad, device=W.device, dtype=W.dtype)], 1) if pad else W
    Xg = X.view(out, -1, G); a = Xg.abs().amax(-1, keepdim=True); best = be = None
    for c in (0.7, 0.8, 0.9, 1.0):
        s = (2*a*c)/(2**k-1); s = torch.where(s == 0, torch.full_like(s, 1e-8), s)
        qv = torch.clamp(torch.round(Xg/s), -(2**(k-1)), 2**(k-1)-1)*s
        er = ((qv-Xg)**2).sum(-1, keepdim=True)
        if best is None: best, be = qv, er
        else: m = er < be; best = torch.where(m, qv, best); be = torch.where(m, er, be)
    return best.view(out, -1)[:, :inn]

def to_inputs(rec, dev):
    sysm = rec.get("system", ""); usr = rec.get("user", rec.get("prompt", ""))
    if A.chat:
        try:                                    # system + user
            msgs = ([{"role": "system", "content": sysm}] if sysm else []) + [{"role": "user", "content": usr}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:                       # template rejects system role (e.g. BioMistral) -> merge into user
            text = tok.apply_chat_template([{"role": "user", "content": (sysm + "\n\n" + usr).strip()}],
                                           tokenize=False, add_generation_prompt=True)
    else:
        text = (sysm + "\n\n" + usr).strip()
    return tok(text, return_tensors="pt").to(dev)

@torch.no_grad()
def decisions(model):
    dev = next(model.parameters()).device
    for d in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(d)
    D, M, topok = [], [], 0; t0 = time.time()
    for i, rec in enumerate(prompts):
        lg = model(**to_inputs(rec, dev)).logits[0, -1]
        if int(lg.argmax()) in ans_ids: topok += 1        # reasoning-model diagnostic
        al = lg[ans_ids].float().cpu().numpy(); o = np.argsort(-al)
        D.append(int(o[0])); M.append(float(al[o[0]] - al[o[1]]))
        if (i+1) % 50 == 0: print(f"    {i+1}/{len(prompts)}", flush=True)
    dt = time.time() - t0
    vram = sum(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())) / 1e9
    return np.array(D), np.array(M), topok/len(prompts), len(prompts)/dt, vram

# ---- run full + each bit-width ----
res = {}
for cfg in ["full"] + bits_list:
    print(f"== {cfg}: loading ({A.quantizer if cfg!='full' else 'bf16'}) ==", flush=True)
    model = load_model(cfg)
    print(f"== {cfg}: decisions ==", flush=True)
    res[cfg] = decisions(model)
    del model; gc.collect(); torch.cuda.empty_cache()

# ---- certificate: Learn-then-Test + bootstrap ----
ref_d = res["full"][0]
def cp_upper(x, n, d):
    if n == 0: return 1.0
    if x >= n: return 1.0
    try:
        from scipy.special import betaincinv
        return float(betaincinv(x + 1, n - x, 1.0 - d))
    except Exception:
        lb = [math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1) for i in range(x+1)]
        def _cdf(p):
            if p <= 0.0: return 1.0
            if p >= 1.0: return 0.0
            lp, l1p = math.log(p), math.log1p(-p)
            t = [lb[i] + i*lp + (n-i)*l1p for i in range(x+1)]
            mx = max(t); return math.exp(mx)*sum(math.exp(v-mx) for v in t)
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo+hi)/2; lo, hi = (mid, hi) if _cdf(mid) > d else (lo, mid)
        return hi
def tau_ltt(margin, flip, alpha, delta, grid):
    cand = np.unique(np.quantile(margin, np.linspace(0, 1, grid)))
    dcorr = delta/len(cand)                                     # Bonferroni over the grid = LTT
    valid = [t for t in cand if cp_upper(int(flip[margin>=t].sum()), int((margin>=t).sum()), dcorr) <= alpha]
    return min(valid) if valid else float("inf")
def cover_valid_over(margin, pres, tau):
    c = margin >= tau; cov = c.mean(); val = pres[c].mean() if c.sum() else 1.0
    return cov, val, cov*val + (1-cov)*1.0
rng = np.random.default_rng(0)

half = int(round(A.cal_frac*len(prompts)))
_floor = 100*(1-(A.delta/A.grid)**(1/max(half,1)))
print(f"\ncalibration={half}  test={len(prompts)-half}  zero-disagreement risk floor ~ {_floor:.2f}% "
      f"(alpha below this is unreachable; raise --cal-frac or n prompts)")
print("\n================ COMPARISON (guarantee {:.0%}, LTT+bootstrap) ================".format(1-A.alpha))
print(f"{'config':>7} {'rawPres':>8} {'certCov(95%CI)':>18} {'validity':>9} {'router':>8} "
      f"{'amortBits':>9} {'size%':>6} {'tok/s':>6} {'VRAM_GB':>8} {'decTok%':>7}")
_, _, tk_f, pps_f, vram_f = res["full"]
print(f"{'full':>7} {100.0:7.1f}% {'-':>18} {'-':>9} {100.0:6.1f}% {16.0:9.2f} {100.0:5.1f} "
      f"{pps_f:6.1f} {vram_f:8.1f} {100*tk_f:6.1f}")
csv_rows = [(A.label, A.quantizer, "full", 100.0, "", "", "", "", 100.0, 16.0, 100.0,
             round(pps_f,2), round(vram_f,2), round(100*tk_f,1))]
for k in bits_list:
    q_d, q_m, tk, pps, vram = res[k]; pres = (q_d == ref_d).astype(int)
    mc, mt, pc, pt = q_m[:half], q_m[half:], pres[:half], pres[half:]
    tau = tau_ltt(mc, (1-pc).astype(bool), A.alpha, A.delta, A.grid)
    cov, val, over = cover_valid_over(mt, pt, tau)
    # bootstrap CIs on the test split
    bc, bv, bo = [], [], []
    for _ in range(A.n_boot):
        idx = rng.integers(0, len(mt), len(mt))
        c2, v2, o2 = cover_valid_over(mt[idx], pt[idx], tau); bc.append(c2); bv.append(v2); bo.append(o2)
    lo, hi = np.percentile(bc, 2.5), np.percentile(bc, 97.5)
    vlo = np.percentile(bv, 2.5); olo = np.percentile(bo, 2.5)
    amort = cov*k + (1-cov)*16
    print(f"{str(k)+'-bit':>7} {100*pres.mean():7.1f}% "
          f"{100*cov:6.1f} [{100*lo:.0f}-{100*hi:.0f}]".rjust(18) +
          f" {100*val:8.1f}% {100*over:6.1f}% {amort:9.2f} {100*amort/16:5.1f} "
          f"{pps:6.1f} {vram:8.1f} {100*tk:6.1f}")
    csv_rows.append((A.label, A.quantizer, f"{k}-bit", round(100*pres.mean(),1), round(100*cov,1),
                     round(100*lo,1), round(100*hi,1), round(100*val,1), round(100*over,1),
                     round(amort,2), round(100*amort/16,1), round(pps,2), round(vram,2), round(100*tk,1)))
print("router = certified@k-bit + uncertified escalated to full precision")
print("decTok% = fraction whose top token is a decision token (low => reasoning model, read unreliable)")
if A.csv:
    import os, csv as _csv
    newf = not os.path.exists(A.csv)
    with open(A.csv, "a", newline="") as f:
        w = _csv.writer(f)
        if newf: w.writerow(["model","quantizer","config","rawPreserve","certCov","cov_lo","cov_hi",
                             "validity","routerOverall","amortBits","sizePctBF16","promptsPerSec","peakVRAM_GB","decTokPct"])
        for r in csv_rows: w.writerow(r)
    print(f"appended {len(csv_rows)} rows -> {A.csv}")
np.savez(A.out, ref_d=ref_d, **{f"d{k}": res[k][0] for k in bits_list}, **{f"m{k}": res[k][1] for k in bits_list})
print(f"saved -> {A.out}")
