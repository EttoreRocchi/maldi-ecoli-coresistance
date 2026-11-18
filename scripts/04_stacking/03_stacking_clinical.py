"""Clinical and calibration profile of the stacked model at Sant'Orsola (logistic regression), against binary relevance
and the chain ensemble on the pooled out-of-fold predictions of the same 15 folds.

Inputs: results/nested_stacking/s_orsola_logistic/r<r>_f<f>.npz (y, br, stack) from scripts/04_stacking/01_nested_stacking.py, and the stored chain
ensemble from results/s_orsola/logistic/cv_results.pkl. Metrics: AUROC, MCC, VME and ME at 0.5, Brier skill score,
calibration slope and intercept, net benefit at threshold probabilities 0.20 to 0.60 and at the prevalence. Isolate-level
paired bootstrap (2,000 resamples; an isolate's three repeats move together) for the differences stack minus BR and stack
minus chain ensemble. Outputs: results/stats/stacking_clinical.json, tables/stacking_clinical.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.config import Config  # noqa: E402
from mecor.cvio import load_cv_results  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402

B = 2000; THRS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]


def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def net_benefit(y, p, t):
    yhat = p >= t; tp = np.sum(yhat & (y == 1)); fp = np.sum(yhat & (y == 0)); n = len(y)
    return tp / n - fp / n * t / (1 - t)


def profile(y, p):
    yhat = (p >= 0.5).astype(int); prev = y.mean()
    out = {"AUROC": roc_auc_score(y, p), "MCC": matthews_corrcoef(y, yhat), "VME": float(np.mean(yhat[y == 1] == 0)), "ME": float(np.mean(yhat[y == 0] == 1)),
           "Brier_skill": 1 - np.mean((p - y) ** 2) / np.mean((prev - y) ** 2)}
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y); out["cal_slope"] = float(lr.coef_[0, 0]); out["cal_intercept"] = float(lr.intercept_[0])
    for t in THRS: out[f"NB_{t:.2f}"] = net_benefit(y, p, t)
    out["NB_prev"] = net_benefit(y, p, prev); out["NB_treat_all_prev"] = prev - (1 - prev) * prev / (1 - prev)
    return out


def main():
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="s_orsola_logistic"); args = ap.parse_args()  # noqa: I001
    cfg = Config.from_yaml(str(ROOT / "configs/s_orsola.yaml")); cfg.model = "logistic"; X, Y = load_dataset(cfg); labels = list(Y.columns)
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    folds = {(g // 5, g % 5): (tr, te) for g, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42).split(X, strat))}
    cv = load_cv_results(ROOT / "results/s_orsola/logistic/cv_results.pkl"); ens = {(fr.repeat_idx, fr.fold_idx): fr for fr in cv.fold_results if fr.method == "ensemble"}
    iso, y, P = [], {d: [] for d in labels}, {m: {d: [] for d in labels} for m in ("br", "ens", "stack")}
    for (r, f), (tr, te) in sorted(folds.items()):
        z = np.load(ROOT / f"results/nested_stacking/{args.tag}/r{r}_f{f}.npz"); fr = ens[(r, f)]
        assert np.array_equal(z["y"][:, 0], np.asarray(fr.y_true[labels[0]]).astype(int))
        iso.append(te)
        for j, d in enumerate(labels):
            y[d].append(z["y"][:, j]); P["br"][d].append(z["br"][:, j]); P["stack"][d].append(z["stack"][:, j]); P["ens"][d].append(np.asarray(fr.y_prob[d]))
    iso = np.concatenate(iso); y = {d: np.concatenate(v) for d, v in y.items()}; P = {m: {d: np.concatenate(v) for d, v in dd.items()} for m, dd in P.items()}
    rng = np.random.default_rng(42); n_iso = len(Y); out = {"point": {}, "contrasts": {}}
    lines = [f"# Clinical profile of the stacked model ({args.tag}; pooled out-of-fold predictions of the 15 folds)", ""]
    for d in labels:
        out["point"][d] = {m: profile(y[d], P[m][d]) for m in ("br", "ens", "stack")}
        boot = {("stack", "br"): [], ("stack", "ens"): []}
        for _ in range(B):
            pick = rng.integers(0, n_iso, n_iso); W = np.bincount(pick, minlength=n_iso)[iso]; idx = np.repeat(np.arange(len(iso)), W)
            yy = y[d][idx]
            if len(np.unique(yy)) < 2: continue
            pr = {m: profile(yy, P[m][d][idx]) for m in ("br", "ens", "stack")}
            for a, b in boot: boot[(a, b)].append({k: pr[a][k] - pr[b][k] for k in pr[a]})
        out["contrasts"][d] = {}
        for (a, b), lst in boot.items():
            out["contrasts"][d][f"{a}-{b}"] = {k: {"delta": out["point"][d][a][k] - out["point"][d][b][k], "ci": [float(np.percentile([x[k] for x in lst], 2.5)), float(np.percentile([x[k] for x in lst], 97.5))], "p_boot": float(np.mean(np.array([x[k] for x in lst]) <= 0))} for k in lst[0]}
        lines += [f"## {d}", "", "| Metric | BR | chain ensemble | stacking | stack minus BR [95% CI] | stack minus ensemble [95% CI] |", "|---|---|---|---|---|---|"]
        for k in ["AUROC", "MCC", "VME", "ME", "Brier_skill", "cal_slope", "cal_intercept", "NB_0.30", "NB_0.40", "NB_0.50", "NB_0.60", "NB_prev"]:
            c1, c2 = out["contrasts"][d]["stack-br"][k], out["contrasts"][d]["stack-ens"][k]
            lines.append(f"| {k} | {out['point'][d]['br'][k]:.3f} | {out['point'][d]['ens'][k]:.3f} | {out['point'][d]['stack'][k]:.3f} | {c1['delta']:+.3f} [{c1['ci'][0]:+.3f}, {c1['ci'][1]:+.3f}] | {c2['delta']:+.3f} [{c2['ci'][0]:+.3f}, {c2['ci'][1]:+.3f}] |")
        lines.append(f"\nTreat-all net benefit at the prevalence ({y[d].mean():.3f}): {out['point'][d]['br']['NB_treat_all_prev']:.3f}\n")
    json.dump(out, open(ROOT / f"results/stats/stacking_clinical_{args.tag}.json", "w"), indent=1, default=float); (ROOT / f"tables/stacking_clinical_{args.tag}.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
