"""Nested stacking across sites and learners, with a permuted-probability control and saved out-of-fold predictions.

For each outer fold (the pipeline's 5 x 3 repeated stratified folds, seed 42), the binary-relevance probabilities of the
three drugs are cross-fitted (5-fold) inside the training fold, a logistic meta-model per drug is fitted on their logits
and applied to the binary-relevance probabilities of the test fold. Learners: logistic regression (liblinear L1/L2 grid), or the saved CatBoost / TabPFN binary-relevance estimators (best fold parameters, refitted).
Control: the meta-model applied to test probabilities whose two other columns are permuted across isolates (mean of 20
permutations): same meta-model, no cross-label information. Reference: the stored chain ensemble of the same fold when
the cv_results pickle can be loaded.
Option --with-spectrum (logistic only): also fit the X+P variant, a logistic model with the same grid on the spectrum plus
the three cross-fitted logits, saved per fold as r<r>_f<f>_xp.json.
Outputs: results/nested_stacking/<site>_<model>/r<r>_f<f>.json and .npz (y, br, stack, stack_perm),
         results/stats/nested_stacking_<site>_<model>.json, tables/nested_stacking_<site>_<model>.md
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.config import Config  # noqa: E402
from mecor.cvio import bca_mean_ci, load_cv_results  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402
from mecor.evaluation.checkpoint import atomic_json_dump, atomic_savez, digest, ensure_signature, load_json_or_none, npz_ok  # noqa: E402
from mecor.evaluation.cv import _build_search_estimator  # noqa: E402
from mecor.evaluation.grouping import check_no_patient_overlap, inner_splits, load_patient_groups, outer_splits  # noqa: E402
from mecor.models.persistence import load_model  # noqa: E402

N_PERM = 20


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {"AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan, "MCC": float(matthews_corrcoef(y, yhat))}


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def sign_flip_p(d):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d)
    if n == 0: return np.nan
    obs = abs(d.mean())
    # exact over all 2^n sign patterns for the usual 15 folds; Monte Carlo (200,000 patterns, fixed seed) for more folds
    signs = np.array(list(product([-1, 1], repeat=n))) if n <= 16 else np.random.default_rng(0).choice([-1, 1], size=(200000, n))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--site", required=True); ap.add_argument("--model", default="logistic", choices=["logistic", "catboost", "tabpfn"]); ap.add_argument("--n-jobs", type=int, default=8); ap.add_argument("--grid", default="liblinear", choices=["liblinear", "liblinear_full", "full"], help="hyperparameter grid for the logistic BR and cross-fits (full = the main-analysis grid; liblinear_full = the same grid without saga)")
    ap.add_argument("--with-spectrum", action="store_true", help="also fit the X+P variant (spectrum plus the three logits; logistic only)")
    ap.add_argument("--save-inner", action="store_true", help="also save the cross-fitted training-fold probabilities (r<r>_f<f>_inner.npz: P_train, Y_train, train and test indices) for the training-only margin; folds already computed are recomputed and checked against the stored predictions")
    ap.add_argument("--folds", default=None, help="comma-separated fold keys r<r>_f<f> to compute (parallel workers); the summary is written only by a run without --folds")
    args = ap.parse_args(); site, model = args.site, args.model
    cfg = Config.from_yaml(str(ROOT / f"configs/{site}.yaml")); cfg.model = model; cfg.n_jobs = args.n_jobs
    if model == "logistic": cfg.param_search_mode = args.grid
    tag = f"{site}_{model}" + ("" if (model != "logistic" or args.grid == "liblinear") else f"_{args.grid}")
    X, Y = load_dataset(cfg); labels = list(Y.columns); Xa = np.asarray(X, float)
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    groups = load_patient_groups(cfg, X.index)   # None unless the site config groups by patient (driams_a_grouped)
    if groups is None:
        folds = {(g // 5, g % 5): (tr, te) for g, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=cfg.n_repeats_outer, random_state=42).split(X, strat))}
    else:
        if model != "logistic": raise SystemExit("patient-grouped stacking is implemented for logistic regression only")
        splits = outer_splits(strat.values, groups, 5, cfg.n_repeats_outer, 42); check_no_patient_overlap(splits, groups)
        folds = {(g // 5, g % 5): (tr, te) for g, (tr, te) in enumerate(splits)}
    ens_ref = None
    try:
        cv = load_cv_results(ROOT / f"results/{site}/{model}/cv_results.pkl")
        ens_ref = {(fr.repeat_idx, fr.fold_idx): fr for fr in cv.fold_results if fr.method == "ensemble"}
        for (r, f), fr in ens_ref.items():
            assert np.array_equal(np.asarray(fr.y_true[labels[0]]).astype(int), Y.iloc[folds[(r, f)][1]][labels[0]].values.astype(int)), "fold reconstruction mismatch"
        print("chain ensemble reference loaded and folds verified", flush=True)
    except Exception as e:  # TabPFN pickles need numpy 2; the ensemble reference then comes from results/stats
        print("ensemble reference unavailable:", repr(e)[:200], flush=True)
    model_dir = ROOT / f"results/{site}/models"
    out_dir = ROOT / "results/nested_stacking" / tag; out_dir.mkdir(parents=True, exist_ok=True)
    # checkpoints are valid only for these folds and settings: refuse to resume with anything else
    ensure_signature(out_dir, {"site": site, "model": model, "grid": args.grid if model == "logistic" else "stored",
                               "group_by_patient": groups is not None, "test_folds": digest(*[folds[k][1] for k in sorted(folds)])})
    per_fold = {}
    only = set(args.folds.split(",")) if args.folds else None
    for (r, f), (tr, te) in sorted(folds.items()):
        if only is not None and f"r{r}_f{f}" not in only: continue
        fp, fz, fx = out_dir / f"r{r}_f{f}.json", out_dir / f"r{r}_f{f}.npz", out_dir / f"r{r}_f{f}_xp.json"
        rec_done = load_json_or_none(fp); xp_done = load_json_or_none(fx) if args.with_spectrum else None
        need_main = not (rec_done is not None and npz_ok(fz)); need_xp = args.with_spectrum and xp_done is None
        fi = out_dir / f"r{r}_f{f}_inner.npz"; need_inner = args.save_inner and not npz_ok(fi)
        if not (need_main or need_xp or need_inner):
            if ens_ref is not None and f"ens_{labels[0]}" not in rec_done:   # the chain-ensemble reference became available after this fold was computed
                fr = ens_ref[(r, f)]
                for d in labels: rec_done[f"ens_{d}"] = metrics(np.asarray(fr.y_true[d]), np.asarray(fr.y_prob[d]))
                atomic_json_dump(rec_done, fp)
            per_fold[(r, f)] = rec_done
            if args.with_spectrum: per_fold[(r, f)].update(xp_done)
            continue
        Xtr, Xte = Xa[tr], Xa[te]; Ptr = np.zeros((len(tr), 3)); Pte = np.zeros((len(te), 3))
        g_tr = None if groups is None else groups[tr]; s_tr = strat.values[tr]
        for j, d in enumerate(labels):
            ytr = Y[d].values[tr]
            if g_tr is None:
                if model == "logistic": est = _build_search_estimator("logistic", cfg)
                else:
                    base = load_model(model_dir, "binary_relevance", r, f, model).estimators_[d]
                    est = clone(base.best_estimator_) if hasattr(base, "best_estimator_") else clone(base)
                Ptr[:, j] = cross_val_predict(clone(est), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=42), method="predict_proba")[:, 1]
            else:   # patient-grouped cross-fitting, each fit with patient-grouped inner folds
                for a, b in inner_splits(s_tr, g_tr, 5, 42):
                    est_k = _build_search_estimator("logistic", cfg, inner_splits(s_tr[a], g_tr[a], cfg.n_splits_inner, cfg.seed))
                    est_k.fit(Xtr[a], ytr[a]); Ptr[b, j] = est_k.predict_proba(Xtr[b])[:, 1]
                est = _build_search_estimator("logistic", cfg, inner_splits(s_tr, g_tr, cfg.n_splits_inner, cfg.seed))
            est.fit(Xtr, ytr); Pte[:, j] = est.predict_proba(Xte)[:, 1]
        rec = {}; preds = {"y": Y.iloc[te].values.astype(int), "br": Pte.copy(), "stack": np.zeros_like(Pte), "stack_perm": np.zeros_like(Pte)}
        rng = np.random.default_rng(42 + 100 * r + f)
        for j, d in enumerate(labels):
            yte = Y[d].values[te]; rec[f"br_{d}"] = metrics(yte, Pte[:, j])
            meta_cv = StratifiedKFold(3, shuffle=True, random_state=42) if g_tr is None else inner_splits(s_tr, g_tr, 3, 42)
            meta = GridSearchCV(LogisticRegression(max_iter=2000), {"C": [0.01, 0.1, 1.0, 10.0]}, cv=meta_cv, scoring="roc_auc")
            meta.fit(logit(Ptr), Y[d].values[tr]); ps = meta.predict_proba(logit(Pte))[:, 1]; preds["stack"][:, j] = ps; rec[f"stack_{d}"] = metrics(yte, ps)
            others = [k for k in range(3) if k != j]; acc = np.zeros(len(te))
            for _ in range(N_PERM):
                Pp = Pte.copy(); perm = rng.permutation(len(te)); Pp[:, others] = Pte[perm][:, others]; acc += meta.predict_proba(logit(Pp))[:, 1]
            pp = acc / N_PERM; preds["stack_perm"][:, j] = pp; rec[f"stack_perm_{d}"] = metrics(yte, pp)
            if ens_ref is not None:
                fr = ens_ref[(r, f)]; rec[f"ens_{d}"] = metrics(np.asarray(fr.y_true[d]), np.asarray(fr.y_prob[d]))
        if need_inner:   # training-only margin inputs; when the fold was already stored, the recomputation must reproduce it
            if not need_main:
                old = np.load(fz)
                assert np.allclose(old["br"], preds["br"], atol=1e-6) and np.allclose(old["stack"], preds["stack"], atol=1e-6), f"r{r} f{f}: recomputed fold differs from the stored one"
            atomic_savez(fi, P_train=Ptr, Y_train=Y.iloc[tr].values.astype(int), train_idx=np.asarray(tr), test_idx=np.asarray(te))
            print(f"{site} {model} fold r{r} f{f}: inner probabilities saved", flush=True)
        if need_main:   # npz first, json last: the json marks the fold as complete
            atomic_savez(fz, **preds); atomic_json_dump(rec, fp)
            print(f"{site} {model} fold r{r} f{f} done: " + ", ".join(f"{d} BR {rec['br_'+d]['AUROC']:.3f} stack {rec['stack_'+d]['AUROC']:.3f}" for d in labels), flush=True)
        per_fold[(r, f)] = load_json_or_none(fp)
        if need_xp:   # X+P: spectrum plus the three cross-fitted logits, same logistic grid as the BR models
            rec_xp = {}
            for j, d in enumerate(labels):
                est2 = _build_search_estimator("logistic", cfg, None if g_tr is None else inner_splits(s_tr, g_tr, cfg.n_splits_inner, cfg.seed))
                est2.fit(np.column_stack([Xtr, logit(Ptr)]), Y[d].values[tr])
                rec_xp[f"stack_xp_{d}"] = metrics(Y[d].values[te], est2.predict_proba(np.column_stack([Xte, logit(Pte)]))[:, 1])
            atomic_json_dump(rec_xp, fx); print(f"{site} {model} fold r{r} f{f} X+P done", flush=True)
        if args.with_spectrum: per_fold[(r, f)].update(load_json_or_none(fx))
    if only is not None: return   # worker run: checkpoints only
    keys = sorted(per_fold); methods = ["stack", "stack_perm"] + (["ens"] if all(f"ens_{labels[0]}" in per_fold[k] for k in keys) else []) + (["stack_xp"] if args.with_spectrum else [])
    out = {"site": site, "model": model, "n_folds": len(keys), "rows": []}
    lines = [f"# Nested stacking, {site}, {model}, {args.grid if model == 'logistic' else 'stored'} grid ({len(keys)} outer folds; meta-logistic on the three cross-fitted BR probabilities)", "",
             "| Drug | BR AUROC | stack AUROC | dAUROC stack vs BR [CI] p | dMCC stack vs BR (p) | stack vs permuted-others [CI] p | " + ("stack vs chain ensemble [CI] p | chain ensemble vs BR [CI] p |" if "ens" in methods else ""),
             "|---|---|---|---|---|---|" + ("---|---|" if "ens" in methods else "")]
    for d in labels:
        A = {m: np.array([per_fold[k][f"{m}_{d}"]["AUROC"] for k in keys]) for m in ["br"] + methods}
        M = {m: np.array([per_fold[k][f"{m}_{d}"]["MCC"] for k in keys]) for m in ["br"] + methods}
        def contrast(a, b):
            dd = A[a] - A[b]; _, lo, hi = bca_mean_ci(dd); return {"delta": float(np.nanmean(dd)), "ci": [float(lo), float(hi)], "p": sign_flip_p(dd), "dMCC": float(np.nanmean(M[a] - M[b])), "p_mcc": sign_flip_p(M[a] - M[b])}
        row = {"drug": d, "br_auroc": float(A["br"].mean()), "stack_auroc": float(A["stack"].mean()), "stack_vs_br": contrast("stack", "br"), "stack_vs_perm": contrast("stack", "stack_perm")}
        if "ens" in methods: row["ens_auroc"] = float(A["ens"].mean()); row["stack_vs_ens"] = contrast("stack", "ens"); row["ens_vs_br"] = contrast("ens", "br")
        if "stack_xp" in methods: row["stack_xp_auroc"] = float(A["stack_xp"].mean()); row["stack_xp_vs_br"] = contrast("stack_xp", "br"); row["stack_vs_stack_xp"] = contrast("stack", "stack_xp")
        out["rows"].append(row)
        def fmt(c): return f"{c['delta']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] {c['p']:.3f}"
        line = f"| {d} | {row['br_auroc']:.3f} | {row['stack_auroc']:.3f} | {fmt(row['stack_vs_br'])} | {row['stack_vs_br']['dMCC']:+.3f} ({row['stack_vs_br']['p_mcc']:.3f}) | {fmt(row['stack_vs_perm'])} |"
        if "ens" in methods: line += f" {fmt(row['stack_vs_ens'])} | {fmt(row['ens_vs_br'])} |"
        lines.append(line)
    if args.with_spectrum:
        lines += ["", "| Drug | stack X+P AUROC | X+P vs BR [CI] p | stack vs X+P [CI] p |", "|---|---|---|---|"]
        lines += [f"| {r_['drug']} | {r_['stack_xp_auroc']:.3f} | {fmt(r_['stack_xp_vs_br'])} | {fmt(r_['stack_vs_stack_xp'])} |" for r_ in out["rows"]]
    atomic_json_dump(out, ROOT / f"results/stats/nested_stacking_{tag}.json", indent=1)
    (ROOT / f"tables/nested_stacking_{tag}.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__":
    main()
