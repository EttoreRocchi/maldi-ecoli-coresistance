"""Diversity-matched null for the chain ensemble: the six chain orderings trained on PERMUTED upstream labels.

Each null chain has the structure of the real one (first link = binary relevance of the first label; downstream links fit
on the spectrum plus the upstream label columns; hard upstream predictions passed at test), but the upstream label columns
of the training isolates are permuted jointly across isolates (one permutation per fold and ordering, fixed seed). The
null keeps the extra columns, the six-member averaging and the tuning, and removes the information from the other drugs.
Real chains and binary relevance come from the stored nested-CV run of the same config (logistic, same folds); the first
fold is also refitted with true upstream labels to check this implementation against the stored chains.
Usage: --site s_orsola [--folds r0_f0,...] [--n-jobs N]; a run without --folds writes the summary:
results/stats/chain_diversity_null_<site>.json, tables/chain_diversity_null_<site>.md
"""
from __future__ import annotations

import argparse
import json
from itertools import permutations, product
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.config import Config  # noqa: E402
from mecor.cvio import bca_mean_ci, load_cv_results  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402
from mecor.evaluation.cv import _build_search_estimator  # noqa: E402


def fit_chain(Xtr, Ytr, Xte, order, cfg, permute_seed=None, first_cache=None):
    """Chain with true-label (or permuted-label) training of the links and hard passing at test. Returns {label: prob}."""
    n = len(Xtr); perm = None if permute_seed is None else np.random.default_rng(permute_seed).permutation(n)
    U_tr, U_te, out = [], [], {}
    for k, d in enumerate(order):
        if k == 0 and first_cache is not None and d in first_cache:
            est = first_cache[d]
        else:
            Ztr = Xtr if not U_tr else np.column_stack([Xtr] + U_tr)
            est = _build_search_estimator("logistic", cfg); est.fit(Ztr, Ytr[d].values)
            if k == 0 and first_cache is not None: first_cache[d] = est
        Zte = Xte if not U_te else np.column_stack([Xte] + U_te)
        p = est.predict_proba(Zte)[:, 1]; out[d] = p
        up = Ytr[d].values.astype(float)
        U_tr.append(up if perm is None else up[perm]); U_te.append((p >= 0.5).astype(float))
    return out


def setup(site):
    cfg = Config.from_yaml(str(ROOT / f"configs/{site}.yaml")); cfg.model = "logistic"
    X, Y = load_dataset(cfg); labels = list(Y.columns)
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=cfg.n_repeats_outer, random_state=42).split(X, strat))
    return cfg, X, Y, labels, {(g // 5, g % 5): s for g, s in enumerate(splits)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--site", required=True); ap.add_argument("--folds", default=None); ap.add_argument("--n-jobs", type=int, default=16)
    a = ap.parse_args()
    cfg, X, Y, labels, folds = setup(a.site); cfg.n_jobs = a.n_jobs
    Xa = np.asarray(X, float)
    cv = load_cv_results(ROOT / f"results/{a.site}/logistic/cv_results.pkl")
    stored = {(fr.repeat_idx, fr.fold_idx, fr.method): fr for fr in cv.fold_results}
    out_dir = ROOT / "results/chain_diversity_null" / a.site; out_dir.mkdir(parents=True, exist_ok=True)
    only = set(a.folds.split(",")) if a.folds else None
    orders = list(permutations(labels))
    for (r, f), (tr, te) in sorted(folds.items()):
        key = f"r{r}_f{f}"
        if only is not None and key not in only: continue
        fz = out_dir / f"{key}.npz"
        if fz.exists(): continue
        ens_fr = stored[(r, f, "ensemble")]
        assert np.array_equal(np.asarray(ens_fr.y_true[labels[0]]).astype(int), Y.iloc[te][labels[0]].values.astype(int)), "fold mismatch with the stored run"
        Xtr, Xte, Ytr = Xa[tr], Xa[te], Y.iloc[tr].reset_index(drop=True)
        cache = {}; null = {}
        for o_idx, order in enumerate(orders):
            null["_".join(order)] = fit_chain(Xtr, Ytr, Xte, order, cfg, permute_seed=1000 * r + 10 * f + o_idx, first_cache=cache)
        res = {"y": Y.iloc[te].values.astype(int), "null_ens": np.column_stack([np.mean([null[k][d] for k in null], axis=0) for d in labels])}
        if (r, f) == (0, 0):   # implementation check: true-label chains must reproduce the stored chains
            diffs = []
            for order in orders:
                real = fit_chain(Xtr, Ytr, Xte, order, cfg, permute_seed=None, first_cache=cache)
                st = stored[(0, 0, "chain_" + "_".join(order))]
                diffs += [float(np.max(np.abs(real[d] - np.asarray(st.y_prob[d])))) for d in labels]
            res["check_max_abs_diff_vs_stored_chains"] = np.array(max(diffs))
            print(f"implementation check, max |real - stored| = {max(diffs):.2e}", flush=True)
        tmp = fz.with_name(fz.stem + ".tmp.npz"); np.savez(tmp, **res); tmp.rename(fz)
        print(f"{a.site} {key}: null ensemble " + " ".join(f"{d} {roc_auc_score(res['y'][:, i], res['null_ens'][:, i]):.3f}" for i, d in enumerate(labels)), flush=True)
    if only is not None: return

    keys = sorted(folds)
    Z = {k: np.load(out_dir / f"r{k[0]}_f{k[1]}.npz") for k in keys}
    from scipy import stats as st_

    def sign_flip_p(d):
        d = np.asarray(d, float); s = np.array(list(product([-1, 1], repeat=len(d))))
        return float(np.mean(np.abs((s * d).mean(axis=1)) >= abs(d.mean()) - 1e-12))

    def nb_p(d, ratio=0.25):   # Nadeau and Bengio corrected resampled t test
        d = np.asarray(d, float); J = len(d); v = d.var(ddof=1)
        return float(2 * st_.t.sf(abs(d.mean() / np.sqrt((1 / J + ratio) * v)), J - 1)) if v > 0 else float("nan")

    def met(y, p, m):
        return roc_auc_score(y, p) if m == "AUROC" else matthews_corrcoef(y, (p >= 0.5).astype(int))
    out = {"site": a.site, "n_folds": len(keys), "rows": [], "check_max_abs_diff_vs_stored_chains": float(Z[keys[0]]["check_max_abs_diff_vs_stored_chains"])}
    L = [f"# Chain ensemble versus a diversity-matched null (six chains on permuted upstream labels), {a.site}, logistic, {len(keys)} outer folds", "",
         "p: exact sign-flip (15 folds) or Nadeau and Bengio corrected t test (more folds).", "",
         "| Drug | Metric | BR | Null ensemble | Chain ensemble | Null minus BR | Chain minus BR | Chain minus null |", "|---|---|---|---|---|---|---|---|"]
    for i, d in enumerate(labels):
        for m in ("AUROC", "MCC"):
            br = np.array([met(Z[k]["y"][:, i], np.asarray(stored[(k[0], k[1], "binary_relevance")].y_prob[d]), m) for k in keys])
            ce = np.array([met(Z[k]["y"][:, i], np.asarray(stored[(k[0], k[1], "ensemble")].y_prob[d]), m) for k in keys])
            nu = np.array([met(Z[k]["y"][:, i], Z[k]["null_ens"][:, i], m) for k in keys])

            def c(x):
                _, lo, hi = bca_mean_ci(x); p = sign_flip_p(x) if len(x) <= 16 else nb_p(x)
                return {"delta": float(x.mean()), "ci": [float(lo), float(hi)], "p": float(p)}
            row = {"drug": d, "metric": m, "BR": float(br.mean()), "null": float(nu.mean()), "chain": float(ce.mean()),
                   "null_minus_BR": c(nu - br), "chain_minus_BR": c(ce - br), "chain_minus_null": c(ce - nu)}
            out["rows"].append(row)
            fm = lambda q: f"{q['delta']:+.3f} [{q['ci'][0]:+.3f}, {q['ci'][1]:+.3f}] p={q['p']:.3f}"
            L.append(f"| {d} | {m} | {row['BR']:.3f} | {row['null']:.3f} | {row['chain']:.3f} | {fm(row['null_minus_BR'])} | {fm(row['chain_minus_BR'])} | {fm(row['chain_minus_null'])} |")
    L += ["", f"Implementation check (fold r0 f0, true-label chains refitted here versus the stored chains): max absolute probability difference {out['check_max_abs_diff_vs_stored_chains']:.2e}"]
    (ROOT / f"results/stats/chain_diversity_null_{a.site}.json").write_text(json.dumps(out, indent=1))
    (ROOT / f"tables/chain_diversity_null_{a.site}.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
