#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_dump_patch.py
===================

Create a patched COPY of bench_ner.py -> bench_ner_dump.py that ALSO writes a
per-document gold/pred JSONL (via a new --dump flag), for ner_paired_analysis.py.

It makes exactly three changes and nothing else:
  1. adds   ap.add_argument("--dump", ...)
  2. score(model)  ->  score(model, cfg, qz)   and appends a JSONL line per doc
     with {dataset,label,quantizer,config,match,doc,gold,pred}, REUSING the
     predictions already computed by extract() (so bench.csv is unchanged and
     the dump is exactly what was scored).
  3. the call  P, R, F = score(model)  ->  P, R, F = score(model, cfg, qz...)

Your original bench_ner.py is NOT modified.  If any target text is not found
(your file differs from the reviewed version), it fails loudly and you apply
the three edits by hand.

USAGE
    python3 apply_dump_patch.py                      # bench_ner.py -> bench_ner_dump.py
    python3 apply_dump_patch.py SRC.py DST.py        # explicit paths

Then run your usual NER command(s) with the patched copy and a shared --dump:
    python bench_ner_dump.py --model ... --label llama3.3-70b \
        --data .../bc5cdr_disease --quantizer rtn --bits 8,6,4 \
        --max-docs 200 --match overlap --chat --csv bench.csv --dump preds.jsonl
    python bench_ner_dump.py --model ... --label llama3.3-70b \
        --data .../bc5cdr_disease --quantizer bnb --bits 4 \
        --max-docs 200 --match overlap --chat --csv bench.csv --dump preds.jsonl
    # ... repeat for every model and for ncbi_disease, ALL appending to preds.jsonl
    python3 ner_paired_analysis.py --input preds.jsonl --out-prefix certa_ner_paired

Use the SAME --data/--max-docs/--split/--match for every run so document index
`doc` identifies the same document across models and configurations.
"""
import sys
import py_compile

ARG_ANCHOR = ('ap.add_argument("--csv", default="bench.csv"); '
              'ap.add_argument("--show", action="store_true")')
ARG_ADD = (ARG_ANCHOR +
           '\nap.add_argument("--dump", default="", '
           'help="append per-doc gold/pred JSONL for the paired analysis")')

OLD_SCORE = '''def score(model):
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
    return P, R, F'''

NEW_SCORE = '''def score(model, cfg, qz):
    TP = FP = FN = 0
    dumpf = open(A.dump, "a") if A.dump else None
    for i, (text, gold) in enumerate(docs):
        pred = extract(model, text)
        if A.match == "overlap":
            tp = sum(any(p in g or g in p for g in gold) for p in pred)
            fn = sum(not any(p in g or g in p for p in pred) for g in gold)
        else:
            tp = len(pred & gold); fn = len(gold - pred)
        TP += tp; FP += len(pred) - tp; FN += fn
        if dumpf:
            dumpf.write(json.dumps({"dataset": dataset, "label": A.label,
                "quantizer": qz, "config": str(cfg), "match": A.match,
                "doc": i, "gold": sorted(gold), "pred": sorted(pred)}) + "\\n")
            dumpf.flush()
        if (i+1) % 50 == 0: print(f"    {i+1}/{len(docs)}", flush=True)
    if dumpf: dumpf.close()
    P = TP/(TP+FP) if TP+FP else 0.0; R = TP/(TP+FN) if TP+FN else 0.0
    F = 2*P*R/(P+R) if P+R else 0.0
    return P, R, F'''

OLD_CALL = "P, R, F = score(model)"
NEW_CALL = 'P, R, F = score(model, cfg, A.quantizer if cfg != "full" else "full")'


def main():
    src_path = sys.argv[1] if len(sys.argv) > 1 else "bench_ner.py"
    dst_path = sys.argv[2] if len(sys.argv) > 2 else "bench_ner_dump.py"
    src = open(src_path, encoding="utf-8").read()

    for name, old in [("--csv argument line", ARG_ANCHOR),
                      ("score(model) function body", OLD_SCORE),
                      ("P, R, F = score(model) call", OLD_CALL)]:
        if old not in src:
            sys.exit("PATCH FAILED: could not find the %s in %s.\n"
                     "Your bench_ner.py differs from the reviewed version; apply "
                     "the three edits by hand (see the instructions I sent)."
                     % (name, src_path))
    if OLD_SCORE.count("def score") != 1 or src.count(OLD_CALL) != 1:
        sys.exit("PATCH FAILED: expected exactly one score() definition and one "
                 "call; found something else. Apply the edits by hand.")

    out = src.replace(ARG_ANCHOR, ARG_ADD, 1)
    out = out.replace(OLD_SCORE, NEW_SCORE, 1)
    out = out.replace(OLD_CALL, NEW_CALL, 1)
    open(dst_path, "w", encoding="utf-8").write(out)
    py_compile.compile(dst_path, doraise=True)   # syntax check
    print("OK -> wrote %s (original %s untouched); it compiles.\n"
          "Run it exactly like bench_ner.py but add  --dump preds.jsonl ."
          % (dst_path, src_path))


if __name__ == "__main__":
    main()
