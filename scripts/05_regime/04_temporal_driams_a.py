"""Temporal validation within DRIAMS-A (2015 to 2018): leave-one-year-out, binary relevance versus classifier chains
(six orderings and their ensemble), logistic regression (liblinear L1/L2 grid by default).

For each test year the models are trained on the other three years. Besides the realized chain ensemble (hard and
probability passing) the script rebuilds the controls used in the main analysis:
  oracle    : true upstream labels of the test year (ceiling)
  permuted  : the same labels permuted across the test isolates (no information; mean of 20 permutations per ordering)
  bag_boot  : bagged binary relevance, six members on bootstrap resamples of the training rows (best BR parameters)
  bag_sub   : bagged binary relevance, six members on random halves of the m/z bins
  stack     : logistic meta-model on the cross-fitted (5-fold) BR probabilities of the three drugs within the training years
Per-isolate predictions are checkpointed in pred_<year>.npz. Isolate-level bootstrap (2000 resamples) gives per-year
CIs and a year-stratified CI for the mean gain of every contrast.
Outputs: results/stats/temporal_driams_a.json, tables/temporal_driams_a.md
"""
from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.config import Config  # noqa: E402
from mecor.data.loader import _load_driams_from_meta  # noqa: E402
from mecor.evaluation.cv import _build_search_estimator  # noqa: E402

N_MEMBERS = 6
N_PERM = 20
B = 2000
VARIANTS = ["br", "bag_boot", "bag_sub", "permuted", "hard", "proba", "oracle", "stack"]
CONTRASTS = [("hard", "br"), ("proba", "br"), ("oracle", "br"), ("permuted", "br"), ("bag_boot", "br"), ("bag_sub", "br"),
             ("hard", "bag_boot"), ("hard", "bag_sub"), ("hard", "permuted"), ("proba", "bag_sub"), ("proba", "permuted"),
             ("stack", "br"), ("stack", "hard"), ("stack", "proba"), ("stack", "bag_sub"), ("stack", "bag_boot")]


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {"AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan, "MCC": float(matthews_corrcoef(y, yhat))}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def _auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan


def fit_member(params, Xtr, ytr, Xte, seed, mode, rng_seed):
    """One bagged BR member with the best BR parameters (same construction as scripts/02_chain_vs_br/04_bagged_br_control.py)."""
    rng = np.random.default_rng(rng_seed)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed, n_jobs=1,
                             C=params["clf__C"], penalty=params["clf__penalty"], solver=params["clf__solver"])
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    if mode == "bootstrap":
        idx = rng.integers(0, len(ytr), len(ytr))
        while len(np.unique(ytr[idx])) < 2:
            idx = rng.integers(0, len(ytr), len(ytr))
        pipe.fit(Xtr[idx], ytr[idx])
        return pipe.predict_proba(Xte)[:, 1]
    cols = rng.choice(Xtr.shape[1], Xtr.shape[1] // 2, replace=False)
    pipe.fit(Xtr[:, cols], ytr)
    return pipe.predict_proba(Xte[:, cols])[:, 1]


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--grid", default="liblinear", choices=["reduced", "liblinear", "full"]); ap.add_argument("--n-jobs", type=int, default=32); args = ap.parse_args()
    cfg = Config.from_yaml(str(ROOT / "configs/driams_a.yaml")); cfg.model = "logistic"; cfg.param_search_mode = args.grid; cfg.n_jobs = args.n_jobs
    tag = "" if args.grid == "liblinear" else f"_{args.grid}"
    Xs, Ys, years = [], [], []
    for year in cfg.driams_years:
        meta = pd.read_csv(Path(cfg.metadata_path) / year / f"{year}_clean.csv", low_memory=False)
        Xy, Yy = _load_driams_from_meta(meta, str(Path(cfg.binned_dir) / year), cfg); Xs.append(np.asarray(Xy, float)); Ys.append(Yy.reset_index(drop=True)); years += [year] * len(Yy)
    X = np.vstack(Xs); Y = pd.concat(Ys, ignore_index=True); years = np.array(years); labels = list(Y.columns)
    out_dir = ROOT / f"results/temporal_driams_a{tag}"; out_dir.mkdir(parents=True, exist_ok=True)
    per_year = {}
    for yi, ty in enumerate(cfg.driams_years):
        f = out_dir / f"test_{ty}.json"; fp = out_dir / f"pred_{ty}.npz"
        if f.exists() and fp.exists() and all(v in np.load(fp).files for v in VARIANTS): per_year[ty] = json.load(open(f)); continue
        tr, te = np.where(years != ty)[0], np.where(years == ty)[0]
        Xtr, Xte = X[tr], X[te]; rec = {"n_train": int(len(tr)), "n_test": int(len(te)), "prevalence": {d: float(Y[d].values[te].mean()) for d in labels}}
        pred = {v: {} for v in VARIANTS}
        br = {}
        for d in labels:
            est = _build_search_estimator("logistic", cfg); est.fit(Xtr, Y[d].values[tr]); br[d] = est
            pred["br"][d] = est.predict_proba(Xte)[:, 1]; rec[f"br_{d}"] = metrics(Y[d].values[te], pred["br"][d])
        print(f"test year {ty}: BR done", flush=True)
        # cross-fitted stacking: 5-fold out-of-fold BR probabilities inside the training years, logistic meta-model per drug
        Ptr = np.zeros((len(tr), len(labels)))
        for j, d in enumerate(labels):
            Ptr[:, j] = cross_val_predict(_build_search_estimator("logistic", cfg), Xtr, Y[d].values[tr], cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
        Pte = np.column_stack([pred["br"][d] for d in labels])
        for j, d in enumerate(labels):
            meta = GridSearchCV(LogisticRegression(max_iter=2000), {"C": [0.01, 0.1, 1.0, 10.0]}, cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring="roc_auc")
            meta.fit(_logit(Ptr), Y[d].values[tr]); pred["stack"][d] = meta.predict_proba(_logit(Pte))[:, 1]; rec[f"stack_{d}"] = metrics(Y[d].values[te], pred["stack"][d])
        print(f"test year {ty}: stacking done", flush=True)
        # bagged BR controls (six members each, best BR parameters)
        for mi, (mode, key) in enumerate((("bootstrap", "bag_boot"), ("subspace", "bag_sub"))):
            for di, d in enumerate(labels):
                probs = Parallel(n_jobs=min(args.n_jobs, N_MEMBERS))(
                    delayed(fit_member)(br[d].best_params_, Xtr, Y[d].values[tr], Xte, cfg.seed, mode, cfg.seed + 1000 * yi + 100 * di + 10 * mi + i) for i in range(N_MEMBERS))
                pred[key][d] = np.mean(probs, axis=0); rec[f"{key}_{d}"] = metrics(Y[d].values[te], pred[key][d])
        print(f"test year {ty}: bagged BR done", flush=True)
        # chains: six orderings, teacher-forced links; realized (hard, proba), oracle and permuted upstream inputs
        ens = {m: {d: [] for d in labels} for m in ("hard", "proba", "oracle", "permuted")}; rng = np.random.default_rng(cfg.seed + yi)
        for order in permutations(labels):
            links = [br[order[0]]]
            for k in range(1, 3):
                est = _build_search_estimator("logistic", cfg); est.fit(np.column_stack([Xtr] + [Y[order[j]].values[tr] for j in range(k)]), Y[order[k]].values[tr]); links.append(est)
            for mode in ("hard", "proba"):
                cols = []
                for k, d in enumerate(order):
                    p = links[k].predict_proba(np.column_stack([Xte] + cols) if cols else Xte)[:, 1]
                    ens[mode][d].append(p); cols.append(((p >= 0.5).astype(float) if mode == "hard" else p).reshape(-1, 1))
            for k, d in enumerate(order):
                if k == 0:
                    ens["oracle"][d].append(pred["br"][d]); ens["permuted"][d].append(pred["br"][d]); continue
                truth = [Y[order[j]].values[te].astype(float) for j in range(k)]
                ens["oracle"][d].append(links[k].predict_proba(np.column_stack([Xte] + [c.reshape(-1, 1) for c in truth]))[:, 1])
                ens["permuted"][d].append(np.mean([links[k].predict_proba(np.column_stack([Xte] + [c[rng.permutation(len(te))].reshape(-1, 1) for c in truth]))[:, 1] for _ in range(N_PERM)], axis=0))
            print(f"test year {ty}: ordering {order} done", flush=True)
        for mode in ("hard", "proba", "oracle", "permuted"):
            for d in labels:
                pred[mode][d] = np.mean(ens[mode][d], axis=0); rec[f"ens_{mode}_{d}"] = metrics(Y[d].values[te], pred[mode][d])
        np.savez_compressed(fp, y=np.column_stack([Y[d].values[te] for d in labels]).astype(int), **{v: np.column_stack([pred[v][d] for d in labels]) for v in VARIANTS})
        json.dump(rec, open(f, "w"), indent=1); per_year[ty] = rec
    # isolate-level bootstrap: per-year AUROC of every variant on the same resamples, contrasts as differences;
    # year-stratified bootstrap of the mean gain reuses the per-year resamples
    rng = np.random.default_rng(42); preds = {ty: np.load(out_dir / f"pred_{ty}.npz") for ty in per_year}
    A = {}
    for ty, z in preds.items():
        n = len(z["y"]); idx = [rng.integers(0, n, n) for _ in range(B)]
        A[ty] = {v: np.array([[_auc(z["y"][i, j], z[v][i, j]) for j in range(len(labels))] for i in idx]) for v in VARIANTS}
    def ci(g): return (float(np.nanpercentile(g, 2.5)), float(np.nanpercentile(g, 97.5)), float(np.mean(g <= 0)))
    boot, boot_mean = {}, {}
    for a, b in CONTRASTS:
        for j, d in enumerate(labels):
            per = []
            for ty in preds:
                g = A[ty][a][:, j] - A[ty][b][:, j]; boot[(ty, d, a, b)] = ci(g); per.append(g)
            boot_mean[(d, a, b)] = ci(np.nanmean(np.column_stack(per), axis=1))
    def val(rec, v, d): return rec[f"br_{d}"] if v == "br" else rec[f"{v}_{d}"] if (v.startswith("bag") or v == "stack") else rec[f"ens_{v}_{d}"]
    rows = []
    lines = [f"# Leave-one-year-out validation within DRIAMS-A (logistic regression, {args.grid} grid)", "",
             "| Test year | n train | n test | Drug | Prevalence | BR AUROC | Ens AUROC hard (gain [95% CI]) | Ens AUROC proba (gain [95% CI]) | BR MCC | Ens MCC hard (gain) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ty, rec in per_year.items():
        for d in labels:
            b, h, p = rec[f"br_{d}"], rec[f"ens_hard_{d}"], rec[f"ens_proba_{d}"]
            row = {"year": ty, "drug": d, "n_train": rec["n_train"], "n_test": rec["n_test"], "prevalence": rec["prevalence"][d], "br_auroc": b["AUROC"], "br_mcc": b["MCC"],
                   "gain_hard": h["AUROC"] - b["AUROC"], "gain_proba": p["AUROC"] - b["AUROC"], "ens_mcc_hard": h["MCC"], "mcc_gain_hard": h["MCC"] - b["MCC"],
                   "auroc": {v: val(rec, v, d)["AUROC"] for v in VARIANTS},
                   "contrasts": {f"{a}-{bb}": {"delta": val(rec, a, d)["AUROC"] - val(rec, bb, d)["AUROC"], "ci": boot[(ty, d, a, bb)][:2], "p_boot": boot[(ty, d, a, bb)][2]} for a, bb in CONTRASTS}}
            rows.append(row); ch, cp = boot[(ty, d, "hard", "br")], boot[(ty, d, "proba", "br")]
            lines.append(f"| {ty} | {rec['n_train']} | {rec['n_test']} | {d} | {rec['prevalence'][d]:.3f} | {b['AUROC']:.3f} | {h['AUROC']:.3f} ({h['AUROC']-b['AUROC']:+.3f} [{ch[0]:+.3f}, {ch[1]:+.3f}]) | {p['AUROC']:.3f} ({p['AUROC']-b['AUROC']:+.3f} [{cp[0]:+.3f}, {cp[1]:+.3f}]) | {b['MCC']:.3f} | {h['MCC']:.3f} ({h['MCC']-b['MCC']:+.3f}) |")
    lines.append("")
    for d in labels:
        g = [r["gain_hard"] for r in rows if r["drug"] == d]; gp = [r["gain_proba"] for r in rows if r["drug"] == d]; bh, bp = boot_mean[(d, "hard", "br")], boot_mean[(d, "proba", "br")]
        lines.append(f"{d}: mean gain over the four test years, hard {np.mean(g):+.3f} [{bh[0]:+.3f}, {bh[1]:+.3f}] (range {min(g):+.3f} to {max(g):+.3f}; year-stratified bootstrap P(gain <= 0) = {bh[2]:.3f}), probability {np.mean(gp):+.3f} [{bp[0]:+.3f}, {bp[1]:+.3f}] (P = {bp[2]:.3f})")
    lines += ["", "## Controls per test year (AUROC)", "", "| Test year | Drug | BR | bagged BR bootstrap | bagged BR subspace | permuted labels | chain hard | chain proba | oracle | stacking |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ty, rec in per_year.items():
        for d in labels:
            lines.append(f"| {ty} | {d} | " + " | ".join(f"{val(rec, v, d)['AUROC']:.3f}" for v in VARIANTS) + " |")
    lines += ["", "## Mean contrast over the four test years (AUROC difference [95% CI], year-stratified bootstrap P(diff <= 0))", "",
              "| Contrast | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    mean_rows = []
    for a, b in CONTRASTS:
        cells = []
        for d in labels:
            m = float(np.mean([r["contrasts"][f"{a}-{b}"]["delta"] for r in rows if r["drug"] == d])); c = boot_mean[(d, a, b)]
            cells.append(f"{m:+.3f} [{c[0]:+.3f}, {c[1]:+.3f}] P={c[2]:.3f}"); mean_rows.append({"contrast": f"{a}-{b}", "drug": d, "delta": m, "ci": c[:2], "p_boot": c[2]})
        lines.append(f"| {a} vs {b} | " + " | ".join(cells) + " |")
    json.dump({"per_year": rows, "mean_contrasts": mean_rows, "n_members": N_MEMBERS, "n_perm": N_PERM, "n_boot": B}, open(ROOT / f"results/stats/temporal_driams_a{tag}.json", "w"), indent=1)
    (ROOT / f"tables/temporal_driams_a{tag}.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
