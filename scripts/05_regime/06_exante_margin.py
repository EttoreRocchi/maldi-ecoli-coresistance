"""Training-only (ex-ante) cross-prediction margin and the stacking gain on the outer folds.

For every outer fold of every stacking run saved with --save-inner (and the learning-curve runs saved with --inner), the
cross-prediction matrix is computed on the cross-fitted training-fold probabilities that the meta-model is fitted on: the
proxy of a label is chosen there (best other-label classifier) and the inner margin is its AUROC minus that of the label's
own classifier, using training isolates only. On the test fold: the stacking gain (stacked minus own binary-relevance
AUROC), the outer margin with the inner-chosen proxy, and the combination term (stacked minus the better of own and proxy).
The inner margin shares no test isolates with the gain, so its association with the gain is not built in.
Decision rule fixed in advance: stack when the inner margin exceeds 0.02, otherwise keep binary relevance; compared with
always binary relevance and always stacking.
Set: --set main (the stacking runs of the paper and the learning curves saved with --inner).
Outputs: results/stats/exante_margin_<set>.json, tables/exante_margin_<set>.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
DELTA = 0.02
SO, DR = ["CIP", "CTX", "CAZ"], ["CIP", "CRO", "CAZ"]
MAIN = ROOT
SETS = {  # stacking runs: (name, folder, labels, group); learning curves: (name, folder, labels)
    "main": {"stack": [("Sant'Orsola", MAIN / "results/nested_stacking/s_orsola_logistic_full", SO, "site"),
                       ("DRIAMS-A", MAIN / "results/nested_stacking/driams_a_grouped_logistic", DR, "site"),
                       ("DRIAMS-B", MAIN / "results/nested_stacking/driams_b_logistic", DR, "site"),
                       ("DRIAMS-C", MAIN / "results/nested_stacking/driams_c_logistic", DR, "site"),
                       ("DRIAMS-D", MAIN / "results/nested_stacking/driams_d_logistic", DR, "site")],
             "lc": [("Sant'Orsola", MAIN / "results/learning_curve/s_orsola_stack_inner", SO),
                    ("DRIAMS-A", MAIN / "results/learning_curve/driams_a_grouped_stack_inner", DR)]},
}


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan


def fold_rows(Ai, At, S, labels, cond):
    """Ai, At: 3 x 3 cross-prediction AUROC [label i, classifier j] on training (cross-fitted) and test; S: stacked AUROC per label."""
    out = []
    for i, d in enumerate(labels):
        if np.isnan(S[i]) or np.isnan(At[i, i]): continue
        others = [k for k in range(3) if k != i]
        j = max(others, key=lambda k: Ai[i, k])
        m_in = Ai[i, j] - Ai[i, i]; gain = S[i] - At[i, i]
        out.append({**cond, "label": d, "proxy": labels[j], "own_inner": Ai[i, i], "own_test": At[i, i], "inner_margin": m_in,
                    "outer_margin": At[i, j] - At[i, i], "gain": gain, "combination": S[i] - max(At[i, i], At[i, j]),
                    "rule": S[i] if m_in > DELTA else At[i, i], "br": At[i, i], "stack": S[i]})
    return out


def collect(which):
    rows = []
    for name, d, labels, group in SETS[which]["stack"]:
        inner = sorted(d.glob("r*_f*_inner.npz")) if d.exists() else []
        expected = 50 if "10 x 5" in name else 15
        if len(inner) != expected or len(list(d.glob("r*_f?.npz"))) != expected:
            print(f"skip {name}: inner probabilities incomplete ({len(inner)} folds)"); continue
        for fi in inner:
            key = fi.name.replace("_inner.npz", ""); zi = np.load(fi); z = np.load(d / f"{key}.npz")
            assert np.array_equal(zi["test_idx"].shape, (len(z["y"]),))
            Ai = np.array([[auc(zi["Y_train"][:, i], zi["P_train"][:, j]) for j in range(3)] for i in range(3)])
            At = np.array([[auc(z["y"][:, i], z["br"][:, j]) for j in range(3)] for i in range(3)])
            S = np.array([auc(z["y"][:, i], z["stack"][:, i]) for i in range(3)])
            rows += fold_rows(Ai, At, S, labels, {"run": name, "group": group, "n": None, "fold": key, "condition": f"{name}"})
    for name, d, labels in SETS[which]["lc"]:
        if not d.exists(): print(f"skip {name}: not run"); continue
        for fj in sorted(d.glob("n*_draw*.json")):
            if fj.name.endswith("_size.json"): continue
            n = int(fj.name.split("_")[0][1:]); folds = json.load(open(fj))
            for q, rec in enumerate(folds):
                if "inner_auc" not in rec: continue
                assert rec["drugs"] == labels
                Ai, At = np.array(rec["inner_auc"], float), np.array(rec["test_auc"], float)
                S = np.array([np.nan] + [rec[f"stack_{x}"]["AUROC"] for x in labels[1:]], float)   # stacking is fitted for the two downstream labels
                rows += fold_rows(Ai, At, S, labels, {"run": name, "group": "size", "n": n, "fold": f"{fj.stem}_{q}", "condition": f"{name}, {n:,} isolates"})
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="main", choices=list(SETS)); which = ap.parse_args().set
    rows = collect(which)
    if not rows: raise SystemExit("no inner probabilities found")
    conds = {}
    for r in rows: conds.setdefault((r["condition"], r["label"]), []).append(r)
    C = []
    for (c, lab), rs in conds.items():
        g = lambda k: float(np.nanmean([x[k] for x in rs]))
        C.append({"condition": c, "label": lab, "group": rs[0]["group"], "run": rs[0]["run"], "n": rs[0]["n"] or 0, "n_folds": len(rs), "inner_margin": g("inner_margin"), "outer_margin": g("outer_margin"),
                  "gain": g("gain"), "combination": g("combination"), "own_inner": g("own_inner"), "own_test": g("own_test"),
                  "br": g("br"), "stack": g("stack"), "rule": g("rule"), "proxy_majority": max(set(x["proxy"] for x in rs), key=[x["proxy"] for x in rs].count)})
    im = np.array([c["inner_margin"] for c in C]); ga = np.array([c["gain"] for c in C]); own = np.array([c["own_inner"] for c in C])
    sp = stats.spearmanr(im, ga); pe = stats.pearsonr(im, ga)
    # partial correlation of gain with inner margin given the own (inner) AUROC, the headroom
    res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
    pc = stats.pearsonr(res(im, own), res(ga, own)); hr = stats.pearsonr(own, ga)
    fold_im = np.array([r["inner_margin"] for r in rows]); fold_ga = np.array([r["gain"] for r in rows])
    out = {"delta": DELTA, "n_conditions": len(C), "n_fold_rows": len(rows),
           "across_conditions": {"spearman": [float(sp.statistic), float(sp.pvalue)], "pearson": [float(pe.statistic), float(pe.pvalue)],
                                 "partial_given_own_auroc": [float(pc.statistic), float(pc.pvalue)], "own_auroc_vs_gain": [float(hr.statistic), float(hr.pvalue)],
                                 "slope_gain_on_inner_margin": float(np.polyfit(im, ga, 1)[0])},
           "fold_level_pearson": float(np.corrcoef(fold_im, fold_ga)[0, 1]),
           "rule": {"mean_br": float(np.mean([c["br"] for c in C])), "mean_stack": float(np.mean([c["stack"] for c in C])), "mean_rule": float(np.mean([c["rule"] for c in C])),
                    "conditions_rule_beats_br": int(sum(c["rule"] > c["br"] + 1e-9 for c in C)), "conditions_rule_below_stack": int(sum(c["rule"] < c["stack"] - 1e-9 for c in C))},
           "conditions": C}
    (ROOT / f"results/stats/exante_margin_{which}.json").write_text(json.dumps(out, indent=1, default=float))
    L = ["# Training-only cross-prediction margin and the stacking gain on the outer folds", "",
         f"Conditions: {len(C)} (run x label); fold rows: {len(rows)}. Inner margin: proxy chosen and scored on the cross-fitted training-fold probabilities only.", "",
         f"Across conditions: Spearman {sp.statistic:.2f} (p {sp.pvalue:.3g}), Pearson {pe.statistic:.2f}, slope {out['across_conditions']['slope_gain_on_inner_margin']:.2f}; "
         f"partial correlation given the own-classifier AUROC {pc.statistic:.2f} (p {pc.pvalue:.3g}); own AUROC alone versus gain {hr.statistic:.2f}. Fold-level Pearson {out['fold_level_pearson']:.2f}.", "",
         f"Rule (stack if inner margin > {DELTA}, else binary relevance), mean AUROC over conditions: rule {out['rule']['mean_rule']:.3f}, always BR {out['rule']['mean_br']:.3f}, always stacking {out['rule']['mean_stack']:.3f}.", "",
         "| Condition | Label | Folds | Own AUROC (test) | Proxy | Inner margin | Outer margin | Stacking gain | Combination term | Rule minus BR |", "|---|---|---|---|---|---|---|---|---|---|"]
    site_order = ["Sant'Orsola", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D", "DRIAMS-A"]; label_order = ["CIP", "CTX", "CRO", "CAZ"]
    for c in sorted(C, key=lambda c: (c["group"] != "site", site_order.index(c["run"]), c["n"], label_order.index(c["label"]))):
        L.append(f"| {c['condition']} | {c['label']} | {c['n_folds']} | {c['own_test']:.3f} | {c['proxy_majority']} | {c['inner_margin']:+.3f} | {c['outer_margin']:+.3f} | {c['gain']:+.3f} | {c['combination']:+.3f} | {c['rule'] - c['br']:+.3f} |")
    (ROOT / f"tables/exante_margin_{which}.md").write_text("\n".join(L) + "\n"); print("\n".join(L[:8]))


if __name__ == "__main__":
    main()
