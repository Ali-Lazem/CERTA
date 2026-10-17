#!/usr/bin/env python3
"""
bench_ner.py — external-benchmark accuracy of full vs 8/6/4-bit models on gold NER
(NCBI-Disease, BC5CDR-disease). Complements the certificate: the certificate proves
per-case agreement with the full model; this measures correctness against human labels.

For each precision it runs the clinical-verifier NER *generation* prompt over the test docs,
parses the JSON entity output, and scores entity-level P/R/F1 vs gold disease spans.

Auto-detects the dataset format: PubTator (.txt), CoNLL/BIO (token<TAB>tag), or
JSONL ({text,entities} or {tokens,ner_tags}). Use --show to print what it detected.

------------------------------------------------------------------------------
HOW TO RUN  (GPU allocation; 70B needs >=2 H200)
    python bench_ner.py --data /path/to/datasets/bc5cdr_disease --show
    python bench_ner.py --model /path/to/model --label llama3.3-70b \
        --data /path/to/datasets/bc5cdr_disease \
        --bits 8,6,4 --quantizer rtn --max-docs 200 --match overlap --chat --csv bench.csv
------------------------------------------------------------------------------
"""
import argparse, glob, gc, json, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="dataset dir or file")
ap.add_argument("--split", default="test", help="substring to pick the split file (test/dev/train)")
ap.add_argument("--model", default=""); ap.add_argument("--label", default="model")
ap.add_argument("--bits", default="8,6,4"); ap.add_argument("--quantizer", choices=["rtn","bnb"], default="rtn")
ap.add_argument("--group", type=int, default=128)
ap.add_argument("--max-docs", type=int, default=200)
ap.add_argument("--max-new-tokens", type=int, default=384)
ap.add_argument("--match", choices=["exact","overlap"], default="overlap")
ap.add_argument("--gold-type", default="", help="if set, keep only extracted entities whose category contains this (e.g. Diag)")
ap.add_argument("--chat", action="store_true")
ap.add_argument("--csv", default="bench.csv"); ap.add_argument("--show", action="store_true")
ap.add_argument("--debug", type=int, default=0, help="print raw generation + parsed entities for the first N docs")
A = ap.parse_args()
_DBG=[0]

# disease-focused prompt so the output lines up with disease-only gold (BC5CDR/NCBI)
NER_SYS = "You are a biomedical named-entity recognition system for disease and disorder mentions."
NER_USER = ('List every DISEASE or DISORDER mentioned in the text. Exclude chemicals, drugs, genes, '
            'proteins, and procedures. Output ONLY a JSON array of the exact verbatim strings as they '
            'appear in the text, for example ["type 2 diabetes", "acute kidney injury"]. If there are '
            'none, output [].\n\nText:\n{text}\n\nJSON array of disease mentions:')

# ---------------- dataset loading (auto-detect) ----------------
def pick_file(path, split):
    if os.path.isfile(path): return path
    cands = glob.glob(os.path.join(path, f"*{split}*")) or glob.glob(os.path.join(path, "*"))
    cands = [c for c in cands if os.path.isfile(c)]
    if not cands: sys.exit(f"no files in {path}")
    return sorted(cands, key=lambda c: (split not in os.path.basename(c), len(c)))[0]

def norm(s): return re.sub(r"\s+", " ", str(s)).strip().lower()

def load_docs(path, split):
    fp = pick_file(path, split)
    head = open(fp, errors="ignore").read(4000)
    fmt = ("pubtator" if re.search(r"\|[ta]\|", head) else
           "inline" if "<category=" in head or "<category " in head else
           "jsonl" if head.lstrip().startswith("{") else "conll")
    docs = []  # list of (text, set(gold_mention_norm))
    if fmt == "pubtator":
        blocks = open(fp, errors="ignore").read().split("\n\n")
        for b in blocks:
            title = abst = ""; gold = set()
            for ln in b.splitlines():
                if "|t|" in ln: title = ln.split("|t|",1)[1]
                elif "|a|" in ln: abst = ln.split("|a|",1)[1]
                else:
                    p = ln.split("\t")
                    if len(p) >= 5: gold.add(norm(p[3]))
            if title or abst: docs.append(((title+" "+abst).strip(), gold))
    elif fmt == "jsonl":
        for ln in open(fp, errors="ignore"):
            ln = ln.strip()
            if not ln: continue
            r = json.loads(ln)
            if "text" in r and "entities" in r:
                gold = {norm(e.get("text", e.get("mention",""))) for e in r["entities"]}
                docs.append((r["text"], gold))
            elif "tokens" in r:
                toks = r["tokens"]; tags = r.get("ner_tags", r.get("tags", r.get("labels", [])))
                text = " ".join(toks); gold = set(); cur = []
                for t, g in zip(toks, tags):
                    gs = str(g)
                    if gs.startswith("B") or gs in ("1","2"):
                        if cur: gold.add(norm(" ".join(cur)))
                        cur = [t]
                    elif gs.startswith("I") and cur: cur.append(t)
                    else:
                        if cur: gold.add(norm(" ".join(cur))); cur = []
                if cur: gold.add(norm(" ".join(cur)))
                docs.append((text, gold))
    elif fmt == "inline":   # NCBI: "PMID text with <category=\"..\">mention</category> ..."
        for ln in open(fp, errors="ignore"):
            ln = ln.strip()
            if not ln: continue
            m = re.match(r"^\d+\s+(.*)$", ln); body = m.group(1) if m else ln
            gold = {norm(x) for x in re.findall(r'<category\s*=\s*"[^"]*"\s*>(.*?)</category>', body)}
            text = re.sub(r"\s+", " ", re.sub(r"</?category[^>]*>", "", body)).strip()
            if gold: docs.append((text, gold))
    else:  # conll/bio
        toks, tags = [], []
        for ln in open(fp, errors="ignore"):
            ln = ln.rstrip("\n")
            if not ln.strip():
                if toks:
                    text = " ".join(toks); gold = set(); cur = []
                    for t, g in zip(toks, tags):
                        if g.startswith("B"):
                            if cur: gold.add(norm(" ".join(cur)))
                            cur = [t]
                        elif g.startswith("I") and cur: cur.append(t)
                        else:
                            if cur: gold.add(norm(" ".join(cur))); cur = []
                    if cur: gold.add(norm(" ".join(cur)))
                    docs.append((text, gold)); toks, tags = [], []
                continue
            parts = re.split(r"[\t ]+", ln)
            toks.append(parts[0]); tags.append(parts[-1])
    return fp, fmt, [d for d in docs if d[1]]        # keep docs that have gold mentions

fp, fmt, docs = load_docs(A.data, A.split)
if A.show:
    print(f"file={fp}\nformat={fmt}\ndocs_with_gold={len(docs)}")
    for t, g in docs[:2]: print("TEXT:", t[:160], "\nGOLD:", list(g)[:8], "\n---")
    sys.exit(0)
docs = docs[:A.max_docs]
print(f"{A.label} | {fp} [{fmt}] | {len(docs)} docs | quantizer={A.quantizer} bits {A.bits}", flush=True)

# ---------------- model + generative NER ----------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as _tlog; _tlog.set_verbosity_error()   # silence cosmetic warnings
tok = AutoTokenizer.from_pretrained(A.model)

def build(cfg):
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
            if p.dim() == 2:
                out, inn = p.shape; G = A.group; pad = (-inn) % G
                W = p.data.float(); X = torch.cat([W, torch.zeros(out, pad, device=W.device)], 1) if pad else W
                Xg = X.view(out, -1, G); a = Xg.abs().amax(-1, keepdim=True); s = (2*a)/(2**cfg-1)
                s = torch.where(s == 0, torch.full_like(s, 1e-8), s)
                q = (torch.clamp(torch.round(Xg/s), -(2**(cfg-1)), 2**(cfg-1)-1)*s).view(out, -1)[:, :inn]
                p.data.copy_(q.to(p.dtype))
    return m

def extract(model, text):
    user = NER_USER.format(text=text[:3000])
    if A.chat:
        try:
            s = tok.apply_chat_template([{"role":"system","content":NER_SYS},{"role":"user","content":user}],
                                        tokenize=False, add_generation_prompt=True)
        except Exception:   # template rejects system role -> merge into user
            s = tok.apply_chat_template([{"role":"user","content":NER_SYS+"\n\n"+user}],
                                        tokenize=False, add_generation_prompt=True)
    else:
        s = NER_SYS + "\n\n" + user
    ids = tok(s, return_tensors="pt", truncation=True, max_length=4096).to(next(model.parameters()).device)
    out = model.generate(**ids, max_new_tokens=A.max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    ents = set()
    cands = re.findall(r"\[[^\[\]]*\]", gen, re.S)               # flat arrays: robust to surrounding prose / extra brackets
    m = re.search(r"\[.*\]", gen, re.S)                          # greedy fallback for a clean single-array output
    if m: cands.append(m.group(0))
    for cand in cands:
        try: arr = json.loads(cand)
        except Exception: continue
        if not isinstance(arr, list): continue
        for e in arr:
            if isinstance(e, str):                                # array of strings (disease prompt)
                s2 = e
            elif isinstance(e, dict):                             # robustness: array of objects
                if A.gold_type and A.gold_type.lower() not in str(e.get("category","")).lower(): continue
                s2 = e.get("entity", e.get("disease", e.get("text", "")))
            else:
                continue
            if s2 and len(str(s2)) > 1: ents.add(norm(s2))
        if ents: break                                            # first array with content wins
    if A.debug and _DBG[0] < A.debug:
        _DBG[0]+=1
        print(f"\n[DEBUG {_DBG[0]}] RAW GEN (<=400ch):\n{gen[:400]!r}\n[DEBUG] array_found={bool(m)}  n_pred={len(ents)}  PRED={sorted(ents)[:12]}", flush=True)
    return ents

def score(model):
    TP = FP = FN = 0
    for i, (text, gold) in enumerate(docs):
        pred = extract(model, text)
        if A.match == "overlap":
            tp = sum(any(p in g or g in p for g in gold) for p in pred)
            fn = sum(not any(p in g or g in p for p in pred) for g in gold)
        else:
            tp = len(pred & gold); fn = len(gold - pred)
        TP += tp; FP += len(pred) - tp; FN += fn
        if (i+1) % 50 == 0: print(f"    {i+1}/{len(docs)}", flush=True)
    P = TP/(TP+FP) if TP+FP else 0.0; R = TP/(TP+FN) if TP+FN else 0.0
    F = 2*P*R/(P+R) if P+R else 0.0
    return P, R, F

import csv as _csv
dataset = os.path.basename(A.data.rstrip("/"))
newf = not os.path.exists(A.csv)
done=set()
if not newf:
    for _r in _csv.DictReader(open(A.csv)): done.add((_r["model"],_r["dataset"],str(_r["config"])))
print(f"\n{'config':>7} {'P':>7} {'R':>7} {'F1':>7}")
with open(A.csv, "a", newline="") as fcsv:
    w = _csv.writer(fcsv)
    if newf: w.writerow(["model","dataset","quantizer","config","precision","recall","f1"])
    for cfg in ["full"] + [int(b) for b in A.bits.split(",")]:
        if (A.label, dataset, str(cfg)) in done:
            print(f"== {cfg}: skip (already in {A.csv}) =="); continue
        print(f"== {cfg}: build ==", flush=True); model = build(cfg)
        P, R, F = score(model)
        print(f"{str(cfg):>7} {P:7.3f} {R:7.3f} {F:7.3f}")
        w.writerow([A.label, dataset, A.quantizer if cfg != "full" else "full", cfg, round(P,4), round(R,4), round(F,4)])
        fcsv.flush(); del model; gc.collect(); torch.cuda.empty_cache()
print(f"appended -> {A.csv}")
