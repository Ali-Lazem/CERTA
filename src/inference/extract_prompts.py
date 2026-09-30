#!/usr/bin/env python3
"""
extract_prompts.py — build REAL verifier prompts from your clinical-verifier candidate data,
using the verbatim SI templates. Auto-detects task from the fields present:
  NER  : entity + category (+ context)      -> YES/NO
  QAR  : question + answer (+ context)      -> YES/NO
  RE   : head + tail + label (+ context)    -> PASS_STRONG/PASS_WEAK/FAIL

Input: a JSONL where each line is one verifier candidate. Field names are flexible
(context may be context/ctx/sentence/text; head_type/tail_type optional).

RUN:
  python extract_prompts.py --in verifier_candidates.jsonl --task ner --out prompts.jsonl
  # NER/QAR sweep with --answers "YES,NO" ; RE sweep with --answers "PASS_STRONG,PASS_WEAK,FAIL"
"""
import argparse, json, sys
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--out", default="prompts.jsonl")
ap.add_argument("--task", choices=["auto","ner","qar","re"], default="auto")
ap.add_argument("--max", type=int, default=0)
ap.add_argument("--ctx-chars", type=int, default=1500)  # clinical-verifier truncates NER context to 1500
A = ap.parse_args()

NER_SYS = "You are a clinical NER verifier. Reply with one word: YES or NO."
NER_USER = ("Entity: {entity} | Category: {category}\nContext: {ctx}\n\n"
            "Is this a valid, specific clinical entity of the stated category "
            "(not demographic, not vague, not hallucinated)?")
QAR_SYS = "You are a clinical QA auditor. Reply with one word: YES or NO."
QAR_USER = ("Context: {ctx}\nQuestion: {question}\nAnswer: {answer}\n\n"
            "Is this answer directly supported by the context? Reject only if "
            "hallucinated or completely absent from context.")
RE_SYS = "You are a clinical relation verifier. Reply: PASS_STRONG, PASS_WEAK, or FAIL."
RE_USER = ("Context: {ctx}\nRelation: {head} ({head_type}) --[{label}]--> {tail} ({tail_type})\n\n"
           "Is this clinically valid and supported by the context?")

def g(r, *keys, default=""):
    for k in keys:
        if k in r and r[k] not in (None, ""): return r[k]
    return default

def detect(r):
    if "entity" in r and "category" in r: return "ner"
    if "question" in r and "answer" in r: return "qar"
    if ("head" in r or "subject" in r) and ("tail" in r or "object" in r): return "re"
    return None

def build(r, task, ctx):
    uid = g(r, "uid", "patient_id", "id")
    if task == "ner":
        return {"system": NER_SYS, "task": "ner", "uid": uid,
                "user": NER_USER.format(entity=g(r,"entity"), category=g(r,"category"), ctx=ctx)}
    if task == "qar":
        return {"system": QAR_SYS, "task": "qar", "uid": uid,
                "user": QAR_USER.format(ctx=ctx, question=g(r,"question"), answer=g(r,"answer"))}
    if task == "re":
        return {"system": RE_SYS, "task": "re", "uid": uid,
                "user": RE_USER.format(ctx=ctx, head=g(r,"head","subject"),
                       head_type=g(r,"head_type","subject_type",default="Entity"),
                       label=g(r,"label","relation"), tail=g(r,"tail","object"),
                       tail_type=g(r,"tail_type","object_type",default="Entity"))}

out, counts = [], {"ner":0,"qar":0,"re":0}
for line in open(A.inp):
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    task = A.task if A.task != "auto" else detect(r)
    if task is None: continue
    ctx = str(g(r,"context","ctx","sentence","text"))[:A.ctx_chars]
    out.append(build(r, task, ctx)); counts[task] += 1
    if A.max and len(out) >= A.max: break
if not out: sys.exit("no records extracted; check field names (need entity/category, question/answer, or head/tail/label)")
with open(A.out, "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
print(f"wrote {len(out)} prompts -> {A.out}   by task: {counts}")
print("NER/QAR: sweep with --answers \"YES,NO\"   |   RE: --answers \"PASS_STRONG,PASS_WEAK,FAIL\"")
