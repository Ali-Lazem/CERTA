#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ner_paired_analysis.py
======================

Paired human-gold correctness analysis for the CERTA external disease-NER
benchmarks, using the SAME entity extraction and the SAME scoring as
bench_ner.py.  It reads the per-document dump that the patched bench_ner.py
writes (--dump), and reports, for each quantized configuration against the
SAME model's BF16 run on the SAME documents:

  (1) paired differences in entity-level precision, recall and F1
      (config minus BF16), aggregated over models, with DOCUMENT-LEVEL
      bootstrap confidence intervals (documents resampled once per iteration,
      applied to both configurations -> paired bootstrap);

  (2) a recall-side gold-entity correctness transition (pooled over models and
      documents), over the SHARED gold entities of each document -- the clean,
      paired unit under overlap matching:
        preserved-covered : BF16 covers gold entity AND config covers it
        newly-covered     : BF16 misses it  -> config covers it   (corrected)
        lost-coverage     : BF16 covers it   -> config misses it   (regressed)
        stayed-missed     : BF16 misses it  AND config misses it
      with net = newly-covered - lost-coverage; and

  (3) the precision-side change as Delta(false positives) = FP_config - FP_BF16
      (spurious predictions introduced or removed), with a bootstrap CI.

SCORING IS COPIED VERBATIM FROM bench_ner.py so the per-config micro F1 the
script prints reproduces your bench.csv exactly.  In overlap mode:
    TP = #predicted entities that overlap some gold entity   (p in g or g in p)
    FN = #gold entities overlapped by no prediction
    FP = len(pred) - TP
    P  = TP/(TP+FP);  R = TP/(TP+FN);  F1 = 2PR/(P+R)
(This recall denominator mixes a prediction-side TP with a gold-side FN; that
is exactly what bench_ner.py computes, and is replicated here for fidelity.)

Nothing is invented: every number comes from your dumped predictions.

---------------------------------------------------------------------------
INPUT: the dump written by the patched bench_ner.py, one JSON object per line:
    {"dataset":"bc5cdr_disease","label":"llama3.3-70b","quantizer":"rtn",
     "config":"full","match":"overlap","doc":0,
     "gold":["...","..."],"pred":["...","..."]}
- config "full" (any quantizer) -> BF16 reference.
- quantizer/config map to display names:
      full        -> BF16
      rtn, 8/6/4  -> RTN-8 / RTN-6 / RTN-4
      bnb, 4      -> NF4
      bnb, 8      -> INT8
- "doc" is the document index within a dataset (stable across runs because
  bench_ner.py loads documents deterministically); it identifies the SAME
  document across all models and configurations, which the paired, document-
  level bootstrap relies on.
- gold is identical for every config/model of a (dataset, doc); the script
  verifies this and errors if not.

---------------------------------------------------------------------------
USAGE
    python3 ner_paired_analysis.py --selftest                 # verify logic
    python3 ner_paired_analysis.py --input preds.jsonl \
        --bootstrap 1000 --seed 12345 --ci 95 --out-prefix certa_ner_paired

Outputs:
    <prefix>_paired_metrics.tex   LaTeX: aggregated paired P/R/F1 deltas + CIs
    <prefix>_coverage_transitions.tex  LaTeX: gold-coverage transitions + Delta FP
    <prefix>_per_model.csv        per (dataset, model, config) metrics + deltas
    <prefix>_summary.json         machine-readable

REPRODUCE-BEFORE-TRUST: the script prints each config's micro P/R/F1 (mean over
models).  Confirm BF16 / NF4 / RTN-4 match bench.csv (and mean|dF1| ~ 0.045 on
BC5CDR, 0.039 on NCBI for NF4) before using the paired numbers.
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict, OrderedDict

import numpy as np

_WS = re.compile(r"\s+")


def norm(s):
    # identical to bench_ner.py norm()
    return _WS.sub(" ", str(s)).strip().lower()


def config_name(quantizer, config):
    c = str(config)
    if c == "full":
        return "BF16"
    q = str(quantizer).lower()
    if q == "rtn":
        return "RTN-%s" % c
    if q == "bnb":
        return {"4": "NF4", "8": "INT8"}.get(c, "BNB-%s" % c)
    return "%s-%s" % (q, c)


# ----------------------------------------------------------------------------
# Scoring, copied verbatim from bench_ner.py (overlap and exact)
# ----------------------------------------------------------------------------
def doc_counts(gold, pred, match):
    """Return (tp, fp, fn) for one document. gold, pred are sets of norm strings."""
    if match == "overlap":
        tp = sum(any((p in g or g in p) for g in gold) for p in pred)
        fn = sum(not any((p in g or g in p) for p in pred) for g in gold)
    else:  # exact
        tp = len(pred & gold)
        fn = len(gold - pred)
    fp = len(pred) - tp
    return tp, fp, fn


def covered(gold_entity, pred, match):
    """Is a single gold entity recovered by the prediction set?"""
    if match == "overlap":
        return any((p in gold_entity or gold_entity in p) for p in pred)
    return gold_entity in pred


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ----------------------------------------------------------------------------
# Load the dump
# ----------------------------------------------------------------------------
def load(path, reference):
    gold = {}                       # (dataset, doc) -> frozenset
    pred = {}                       # (dataset, model, config, doc) -> set
    match_modes = set()
    datasets, models, configs = OrderedDict(), defaultdict(OrderedDict), OrderedDict()
    docs = defaultdict(OrderedDict)  # dataset -> {doc: None}
    n = 0
    with open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ds = str(r["dataset"])
            lab = str(r["label"])
            cfg = config_name(r.get("quantizer", "full"), r["config"])
            d = int(r["doc"])
            match_modes.add(r.get("match", "overlap"))
            g = frozenset(norm(x) for x in r.get("gold", []))
            p = set(norm(x) for x in r.get("pred", []))
            gkey = (ds, d)
            if gkey in gold and gold[gkey] != g:
                raise SystemExit("Inconsistent gold for dataset=%s doc=%d (line %d)"
                                 % (ds, d, ln))
            gold[gkey] = g
            datasets.setdefault(ds, None)
            docs[ds].setdefault(d, None)
            configs.setdefault(cfg, None)
            # BF16 may be dumped by both rtn and bnb runs: dedupe, verify identical
            pk = (ds, lab, cfg, d)
            if pk in pred and pred[pk] != p:
                raise SystemExit(
                    "Conflicting predictions for %s (greedy decoding should be "
                    "deterministic); line %d" % (str(pk), ln))
            pred[pk] = p
            models[ds].setdefault(lab, None)
            n += 1
    if len(match_modes) != 1:
        print("[warn] dump mixes match modes %s; using the first." % match_modes,
              file=sys.stderr)
    match = sorted(match_modes)[0] if match_modes else "overlap"
    if reference not in configs:
        raise SystemExit("reference %r not among configs %s"
                         % (reference, list(configs)))
    cfg_order = [reference] + [c for c in configs if c != reference]
    print("Loaded %d records | match=%s | datasets=%s | configs=%s"
          % (n, match, list(datasets), cfg_order), file=sys.stderr)
    return (list(datasets), {ds: list(models[ds]) for ds in datasets},
            {ds: list(docs[ds]) for ds in datasets}, cfg_order, match, gold, pred)


# ----------------------------------------------------------------------------
# Build per-document arrays
# ----------------------------------------------------------------------------
def build(ds, models, docs, configs, reference, match, gold, pred):
    D = len(docs)
    cnt = {}            # (model, cfg) -> (tp[D], fp[D], fn[D])
    cov = {}            # (model, cfg) -> list over docs of dict{gold_entity: bool}
    missing = 0
    for m in models:
        for cfg in configs:
            tp = np.zeros(D, np.int64); fp = np.zeros(D, np.int64); fn = np.zeros(D, np.int64)
            covlist = []
            for i, d in enumerate(docs):
                g = gold.get((ds, d), frozenset())
                p = pred.get((ds, m, cfg, d), None)
                if p is None:
                    missing += 1
                    p = set()
                a, b, c = doc_counts(g, p, match)
                tp[i], fp[i], fn[i] = a, b, c
                covlist.append({e: covered(e, p, match) for e in g})
            cnt[(m, cfg)] = (tp, fp, fn)
            cov[(m, cfg)] = covlist
    if missing:
        print("  [warn] %s: %d (model,config,doc) predictions absent -> treated "
              "as empty." % (ds, missing), file=sys.stderr)
    return cnt, cov


# ----------------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------------
def micro(cnt, m, cfg, idx):
    tp, fp, fn = cnt[(m, cfg)]
    return prf(int(tp[idx].sum()), int(fp[idx].sum()), int(fn[idx].sum()))


def agg_delta(cnt, models, cfg, ref, idx):
    dP = dR = dF = dAbs = 0.0
    for m in models:
        p0, r0, f0 = micro(cnt, m, ref, idx)
        p1, r1, f1 = micro(cnt, m, cfg, idx)
        dP += p1 - p0; dR += r1 - r0; dF += f1 - f0; dAbs += abs(f1 - f0)
    k = len(models)
    return dP / k, dR / k, dF / k, dAbs / k


def delta_fp(cnt, models, cfg, ref, idx):
    tot = 0
    for m in models:
        _, fpc, _ = cnt[(m, cfg)]
        _, fpr, _ = cnt[(m, ref)]
        tot += int(fpc[idx].sum()) - int(fpr[idx].sum())
    return tot


def ci(samples, level):
    lo = (100 - level) / 2.0
    a = np.asarray(samples, float)
    return float(np.percentile(a, lo)), float(np.percentile(a, 100 - lo))


# ----------------------------------------------------------------------------
# Coverage transitions (recall side), pooled over models & docs
# ----------------------------------------------------------------------------
def transitions(cov, models, cfg, ref, docs):
    """Return per-doc arrays of (preserved, newly, lost, stayed) summed over models."""
    D = len(docs)
    pres = np.zeros(D, np.int64); newc = np.zeros(D, np.int64)
    lost = np.zeros(D, np.int64); stay = np.zeros(D, np.int64)
    for m in models:
        cref = cov[(m, ref)]; ccfg = cov[(m, cfg)]
        for i in range(D):
            gref = cref[i]; gcfg = ccfg[i]
            for e, rok in gref.items():
                cok = gcfg.get(e, False)
                if rok and cok:
                    pres[i] += 1
                elif (not rok) and cok:
                    newc[i] += 1
                elif rok and (not cok):
                    lost[i] += 1
                else:
                    stay[i] += 1
    return pres, newc, lost, stay


# ----------------------------------------------------------------------------
# Analyse
# ----------------------------------------------------------------------------
def analyse(args):
    datasets, models_by, docs_by, configs, match, gold, pred = load(
        args.input, args.reference)
    comp = [c for c in configs if c != args.reference]
    if args.configs:
        want = [c.strip() for c in args.configs.split(",") if c.strip()]
        comp = [c for c in comp if c in want]
        if not comp:
            raise SystemExit("none of --configs present")

    rng = np.random.default_rng(args.seed)
    summary = {"match": match, "reference": args.reference,
               "bootstrap": args.bootstrap, "ci": args.ci, "seed": args.seed,
               "datasets": {}}
    per_model_rows = []

    for ds in datasets:
        models = models_by[ds]; docs = docs_by[ds]; D = len(docs)
        cnt, cov = build(ds, models, docs, configs, args.reference, match, gold, pred)
        idx_all = np.arange(D)

        print("\n=== %s : per-config micro metrics (mean over %d models) ==="
              % (ds, len(models)))
        print("  %-8s %8s %8s %8s" % ("config", "P", "R", "F1"))
        pm = {}
        for cfg in configs:
            P = np.mean([micro(cnt, m, cfg, idx_all)[0] for m in models])
            R = np.mean([micro(cnt, m, cfg, idx_all)[1] for m in models])
            F = np.mean([micro(cnt, m, cfg, idx_all)[2] for m in models])
            pm[cfg] = (float(P), float(R), float(F))
            print("  %-8s %8.4f %8.4f %8.4f" % (cfg, P, R, F))

        boot = [rng.integers(0, D, D) for _ in range(args.bootstrap)]
        dout = {"n_docs": D, "n_models": len(models),
                "per_config_micro_mean": pm, "configs": {}}

        for cfg in comp:
            dP, dR, dF, dAbs = agg_delta(cnt, models, cfg, args.reference, idx_all)
            bP = np.empty(args.bootstrap); bR = np.empty(args.bootstrap)
            bF = np.empty(args.bootstrap); bA = np.empty(args.bootstrap)
            bFP = np.empty(args.bootstrap)
            for k, idx in enumerate(boot):
                bP[k], bR[k], bF[k], bA[k] = agg_delta(cnt, models, cfg,
                                                       args.reference, idx)
                bFP[k] = delta_fp(cnt, models, cfg, args.reference, idx)
            ciP, ciR, ciF = ci(bP, args.ci), ci(bR, args.ci), ci(bF, args.ci)
            ciA, ciFP = ci(bA, args.ci), ci(bFP, args.ci)

            pres, newc, lost, stay = transitions(cov, models, cfg,
                                                 args.reference, docs)
            Ppres, Pnew, Plost, Pstay = (int(pres.sum()), int(newc.sum()),
                                         int(lost.sum()), int(stay.sum()))
            net = Pnew - Plost
            net_boot = np.array([int(newc[i].sum()) - int(lost[i].sum())
                                 for i in boot])
            ciNet = ci(net_boot, args.ci)
            fp_point = delta_fp(cnt, models, cfg, args.reference, idx_all)

            dout["configs"][cfg] = {
                "delta_precision": [dP, ciP[0], ciP[1]],
                "delta_recall": [dR, ciR[0], ciR[1]],
                "delta_f1": [dF, ciF[0], ciF[1]],
                "mean_abs_delta_f1": [dAbs, ciA[0], ciA[1]],
                "delta_false_positives": [fp_point, ciFP[0], ciFP[1]],
                "coverage_transitions": {
                    "preserved_covered": Ppres, "newly_covered": Pnew,
                    "lost_coverage": Plost, "stayed_missed": Pstay,
                    "net": net, "net_ci": [ciNet[0], ciNet[1]],
                },
            }
            for m in models:
                p0, r0, f0 = micro(cnt, m, args.reference, idx_all)
                p1, r1, f1 = micro(cnt, m, cfg, idx_all)
                per_model_rows.append({
                    "dataset": ds, "model": m, "config": cfg,
                    "F1_BF16": round(f0, 4), "F1_cfg": round(f1, 4),
                    "dP": round(p1 - p0, 4), "dR": round(r1 - r0, 4),
                    "dF1": round(f1 - f0, 4)})
        summary["datasets"][ds] = dout

    write_outputs(args, summary, per_model_rows)
    return summary


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def _ci(v, lo, hi, dec=3):
    return "$%+.*f$ $[%+.*f,\\,%+.*f]$" % (dec, v, dec, lo, dec, hi)


def write_outputs(args, summary, rows):
    pref = args.out_prefix
    L = []
    L.append("% Auto-generated by ner_paired_analysis.py  (do not edit by hand)")
    L.append(f"% Paired human-gold deltas (config - {summary['reference']}), "
             f"mean over models, match={summary['match']},")
    L.append(f"% document-level {summary['ci']:g}% bootstrap CIs, "
             f"{summary['bootstrap']} resamples.")
    L.append("\\begin{tabular}{llcccc}")
    L.append("\\toprule")
    L.append("Benchmark & Config & $\\Delta$P & $\\Delta$R & $\\Delta$F1 & "
             "mean $|\\Delta\\text{F1}|$\\\\")
    L.append("\\midrule")
    for ds, d in summary["datasets"].items():
        first = True
        for cfg, c in d["configs"].items():
            L.append("%s & %s & %s & %s & %s & %s\\\\" % (
                ds if first else "", cfg,
                _ci(*c["delta_precision"]), _ci(*c["delta_recall"]),
                _ci(*c["delta_f1"]), _ci(*c["mean_abs_delta_f1"])))
            first = False
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}")
    open(pref + "_paired_metrics.tex", "w").write("\n".join(L) + "\n")

    T = []
    T.append("% Auto-generated by ner_paired_analysis.py  (do not edit by hand)")
    T.append(f"% Gold-entity coverage transitions vs {summary['reference']} "
             f"(recall side), pooled over models and documents; Delta FP is the")
    T.append(f"% change in false positives (precision side). match={summary['match']}.")
    T.append("\\begin{tabular}{llrrrrrc}")
    T.append("\\toprule")
    T.append("Benchmark & Config & \\makecell{Preserved\\\\covered} & "
             "\\makecell{Newly\\\\covered} & \\makecell{Lost\\\\coverage} & "
             "\\makecell{Stayed\\\\missed} & \\makecell{Net\\\\coverage} & "
             "$\\Delta$FP\\\\")
    T.append("\\midrule")
    for ds, d in summary["datasets"].items():
        first = True
        for cfg, c in d["configs"].items():
            tr = c["coverage_transitions"]
            fp = c["delta_false_positives"]
            T.append("%s & %s & %d & %d & %d & %d & %+d $[%+d,\\,%+d]$ & %+d $[%+d,\\,%+d]$\\\\" % (
                ds if first else "", cfg,
                tr["preserved_covered"], tr["newly_covered"], tr["lost_coverage"],
                tr["stayed_missed"], tr["net"],
                int(round(tr["net_ci"][0])), int(round(tr["net_ci"][1])),
                int(round(fp[0])), int(round(fp[1])), int(round(fp[2]))))
            first = False
        T.append("\\midrule")
    T[-1] = "\\bottomrule"
    T.append("\\end{tabular}")
    open(pref + "_coverage_transitions.tex", "w").write("\n".join(T) + "\n")

    with open(pref + "_per_model.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "model", "config",
                           "F1_BF16", "F1_cfg", "dP", "dR", "dF1"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    json.dump(summary, open(pref + "_summary.json", "w"), indent=2)
    print("\nWrote:\n  %s_paired_metrics.tex\n  %s_coverage_transitions.tex\n"
          "  %s_per_model.csv\n  %s_summary.json" % (pref, pref, pref, pref))


# ----------------------------------------------------------------------------
# Self-test (hand-verified overlap numbers; see module header example)
# ----------------------------------------------------------------------------
def selftest():
    gold = {norm("diabetes"), norm("kidney injury")}
    bf16 = {norm("diabetes mellitus"), norm("cancer")}
    nf4 = {norm("diabetes"), norm("acute kidney injury")}

    # overlap counts
    assert doc_counts(gold, bf16, "overlap") == (1, 1, 1), doc_counts(gold, bf16, "overlap")
    assert doc_counts(gold, nf4, "overlap") == (2, 0, 0), doc_counts(gold, nf4, "overlap")
    p, r, f = prf(*doc_counts(gold, bf16, "overlap"))
    assert abs(p - 0.5) < 1e-12 and abs(r - 0.5) < 1e-12 and abs(f - 0.5) < 1e-12
    p, r, f = prf(*doc_counts(gold, nf4, "overlap"))
    assert (p, r, f) == (1.0, 1.0, 1.0)

    # coverage per gold entity
    assert covered(norm("diabetes"), bf16, "overlap") is True
    assert covered(norm("kidney injury"), bf16, "overlap") is False
    assert covered(norm("kidney injury"), nf4, "overlap") is True

    # transition (one doc, one model): preserved=1 (diabetes), newly=1 (kidney), lost=0, stayed=0
    pres = newc = lost = stay = 0
    for e in gold:
        rok = covered(e, bf16, "overlap"); cok = covered(e, nf4, "overlap")
        if rok and cok: pres += 1
        elif (not rok) and cok: newc += 1
        elif rok and (not cok): lost += 1
        else: stay += 1
    assert (pres, newc, lost, stay) == (1, 1, 0, 0), (pres, newc, lost, stay)

    # delta FP = FP_nf4 - FP_bf16 = 0 - 1 = -1
    assert (doc_counts(gold, nf4, "overlap")[1] - doc_counts(gold, bf16, "overlap")[1]) == -1

    # exact-mode sanity
    assert doc_counts({"a", "b"}, {"a", "c"}, "exact") == (1, 1, 1)
    assert config_name("full", "full") == "BF16"
    assert config_name("rtn", "4") == "RTN-4"
    assert config_name("bnb", "4") == "NF4"
    assert config_name("bnb", "8") == "INT8"
    print("SELF-TEST PASSED: overlap/exact scoring, coverage, transitions, "
          "Delta FP and config naming all match hand-computed values.")


def main():
    ap = argparse.ArgumentParser(
        description="Paired human-gold NER correctness analysis mirroring "
                    "bench_ner.py scoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --selftest first.")
    ap.add_argument("--input", help="per-doc dump JSONL from patched bench_ner.py")
    ap.add_argument("--reference", default="BF16")
    ap.add_argument("--configs", default="",
                    help="comma-separated subset of configs vs reference")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--ci", type=float, default=95.0)
    ap.add_argument("--out-prefix", default="certa_ner_paired")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.input:
        ap.error("--input is required (or use --selftest)")
    analyse(a)


if __name__ == "__main__":
    main()
