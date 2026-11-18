#!/usr/bin/env python3
"""Inference checks for the chain-versus-binary-relevance (BR) comparison
on the Sant'Orsola cohort and for the cross-site transfer experiment (no refitting).

Questions
  - Does the chain gain survive inference at the isolate level, rather than over the
      15 outer folds? (a) paired isolate bootstrap of the pooled out-of-fold predictions,
      (b) per-repeat estimates, (c) paired DeLong tests per repeat, (d) minimum detectable
      paired difference given the observed fold-to-fold dispersion.
  - How large is the gain of the best chain ordering when the ordering is selected on
      held-out repeats (nested, honest selection) instead of in-sample (best of six)?
  - Which of the 18 ensemble-versus-BR tests (3 models x 3 drugs x 2 metrics) survive
      Holm and Benjamini-Hochberg adjustment, with and without the logistic regression CTX
      comparisons in the family (a sensitivity check)?
  - Cross-site: exact paired sign-flip tests over the 20 train-test pairs of the
      ensemble-minus-BR AUROC per drug (BCa interval), and linear mixed models of the
      ensemble transfer AUROC and of the chain gain on the training-site co-resistance
      (phi CIP-cephalosporin, phi cephalosporin-cephalosporin), the absolute train-minus-test
      phi mismatch and the prevalence shift, with a random intercept per test site
      (sensitivity: per train site).
  - Sensitivity table: canonical chain training (links trained on true upstream labels)
      versus the cvsens run (links trained on out-of-fold upstream predictions): BR, Ens and
      Best per drug and metric with paired sign-flip p-values, side by side.

Method
  Outer folds are reconstructed with RepeatedStratifiedKFold(5, 3, random_state=42) on the
  joint label string and verified against the stored y_true of every fold and method, so
  every out-of-fold prediction is tied to an isolate id. Bootstrap: 2000 resamples of the
  187 isolates with replacement (one index matrix shared by all comparisons); a resampled
  isolate contributes its three out-of-fold predictions (one per repeat); metrics are
  computed on the pooled resampled predictions; percentile 95 percent intervals; two-sided
  bootstrap p = 2 x fraction of resamples whose delta has the opposite sign or is zero.
  DeLong: fast DeLong (Sun and Xu 2014) with the covariance of the placement values, paired
  on the same isolates, one test per repeat (repeats share isolates and are not combined).
  MDE: paired t-test, alpha 0.05 two-sided, power 0.80 (statsmodels TTestPower) times the
  observed SD of the per-fold (n = 15) or per-repeat (n = 3) differences. Seed 42.

Outputs
  results/stats/inference_checks.json
  tables/inference_checks.md
  figures/supplementary/inference_isolate_bootstrap.png (+ pdf)
  figures/supplementary/honest_best_ordering.png (+ pdf)

Usage
  python scripts/02_chain_vs_br/03_inference_checks.py              # full computation, tables and figures
  python scripts/02_chain_vs_br/03_inference_checks.py --plot-only  # figures only, from the stored JSON
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)

from mecor import shared as _sig
from mecor.config import Config  # noqa: E402
from mecor.cvio import aligned, bca_mean_ci, best_chain, load_cv_results, metric_by_fold  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402

exact_sign_flip_pvalue = _sig.exact_sign_flip_pvalue        # returns (observed mean, p)
best_chain_max_stat_pvalue = _sig.best_chain_max_stat_pvalue

SEED = 42
N_BOOT = 2000
MODELS = [m for m in ("logistic", "catboost", "tabpfn") if (ROOT / f"results/s_orsola/{m}/cv_results.pkl").exists()]
DRUGS = ["CIP", "CTX", "CAZ"]
METRICS4 = ["AUROC", "MCC", "VME", "ME"]
ORDERINGS = ["chain_CIP_CTX_CAZ", "chain_CIP_CAZ_CTX", "chain_CTX_CIP_CAZ",
             "chain_CTX_CAZ_CIP", "chain_CAZ_CIP_CTX", "chain_CAZ_CTX_CIP"]
SITES = ["s_orsola", "driams_a", "driams_b", "driams_c", "driams_d"]
SITE_FROM_CROSS = {"local": "s_orsola", "driams_a": "driams_a", "driams_b": "driams_b",
                   "driams_c": "driams_c", "driams_d": "driams_d"}
SITE_LABEL = {"s_orsola": "Sant'Orsola", "driams_a": "DRIAMS-A", "driams_b": "DRIAMS-B",
              "driams_c": "DRIAMS-C", "driams_d": "DRIAMS-D"}
PRIMARY = ("logistic", "CTX")
COVS = ["phi_tr_cip_ceph", "phi_tr_ceph_caz", "phi_mismatch", "prev_shift"]
COV_LABEL = {"phi_tr_cip_ceph": "phi(train) CIP-ceph", "phi_tr_ceph_caz": "phi(train) ceph-CAZ",
             "phi_mismatch": "|phi train - phi test|", "prev_shift": "prevalence shift (train - test)",
             "log_n_train": "log10 n_train"}
MODEL_LABEL = {"logistic": "logistic", "catboost": "CatBoost", "tabpfn": "TabPFN"}

STATS_JSON = ROOT / "results/stats/within_site_stats.json"
OUT_JSON = ROOT / "results/stats/inference_checks.json"
OUT_MD = ROOT / "tables/inference_checks.md"
FIG_DIR = ROOT / "figures/supplementary"


def short(o):
    return o.replace("chain_", "").replace("_", "->")


def fp(p):
    return "< 0.001" if p < 0.0005 else f"{p:.3f}"


def auroc(y, p):
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def mcc(y, yhat):
    y = np.asarray(y, bool)
    yhat = np.asarray(yhat, bool)
    tp = float(np.sum(y & yhat))
    tn = float(np.sum(~y & ~yhat))
    fp_ = float(np.sum(~y & yhat))
    fn = float(np.sum(y & ~yhat))
    den = np.sqrt((tp + fp_) * (tp + fn) * (tn + fp_) * (tn + fn))
    return float((tp * tn - fp_ * fn) / den) if den > 0 else 0.0


def all_metrics(y, p):
    y = np.asarray(y, int)
    yhat = (np.asarray(p, float) >= 0.5).astype(int)
    r = y == 1
    s = y == 0
    return {"AUROC": auroc(y, p), "MCC": mcc(y, yhat),
            "VME": float((yhat[r] == 0).mean()) if r.any() else np.nan,
            "ME": float((yhat[s] == 1).mean()) if s.any() else np.nan}


def exact_sign_flip_chunked(d):
    """Exact two-sided paired sign-flip test on the mean, enumerated in chunks (n <= 24)."""
    d = np.asarray(d, float)
    n = len(d)
    obs = float(d.mean())
    total = 1 << n
    chunk = 1 << 16
    cnt = 0
    ar = np.arange(n)
    for start in range(0, total, chunk):
        idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
        signs = 1.0 - 2.0 * ((idx[:, None] >> ar[None, :]) & 1)
        null = signs @ d / n
        cnt += int(np.sum(np.abs(null) >= abs(obs) - 1e-15))
    return obs, cnt / total


def delong_paired(y, p_a, p_b):
    """Fast DeLong (Sun and Xu 2014) for two correlated AUROCs on the same isolates.

    Returns auc_a, auc_b, diff = auc_a - auc_b, its SE from the covariance of the
    placement values, z and the two-sided normal p-value."""
    y = np.asarray(y, bool)
    pos = np.flatnonzero(y)
    neg = np.flatnonzero(~y)
    m, n = len(pos), len(neg)
    order = np.concatenate([pos, neg])
    preds = np.vstack([np.asarray(p_a, float)[order], np.asarray(p_b, float)[order]])
    k = 2
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = rankdata(preds[r, :m])
        ty[r] = rankdata(preds[r, m:])
        tz[r] = rankdata(preds[r])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n          # placement values of the positives
    v10 = 1.0 - (tz[:, m:] - ty) / m    # placement values of the negatives
    S = np.cov(v01) / m + np.cov(v10) / n
    L = np.array([1.0, -1.0])
    var = float(L @ S @ L)
    diff = float(aucs[0] - aucs[1])
    if var <= 1e-18:
        z, p = 0.0, 1.0
    else:
        z = diff / np.sqrt(var)
        p = float(2.0 * norm.sf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": diff,
            "se": float(np.sqrt(max(var, 0.0))), "z": float(z), "p": p}


def mde_paired(sd, n):
    """Minimum detectable paired mean difference (alpha 0.05 two-sided, power 0.80)."""
    from statsmodels.stats.power import TTestPower
    if not np.isfinite(sd) or sd <= 0 or n < 2:
        return float("nan")
    d = TTestPower().solve_power(effect_size=None, nobs=n, alpha=0.05, power=0.80,
                                 alternative="two-sided")
    return float(d * sd)


def boot_pvalue(obs, boot):
    if not np.isfinite(obs) or obs == 0:
        return 1.0
    frac = float(np.mean(boot <= 0)) if obs > 0 else float(np.mean(boot >= 0))
    return float(min(1.0, 2.0 * frac))


def to_native(o):
    if isinstance(o, dict):
        return {str(k): to_native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_native(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return to_native(o.tolist())
    return o


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def reconstruct_folds(X, Y):
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    folds = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for g, (tr, te) in enumerate(outer.split(X, strat)):
            folds[(g // 5, g % 5)] = np.asarray(te)
    for r in range(3):
        cnt = np.zeros(len(X), int)
        for f in range(5):
            cnt[folds[(r, f)]] += 1
        assert cnt.min() == 1 and cnt.max() == 1, "each isolate must be tested once per repeat"
    return folds


def verify_folds(cv, folds, Y):
    n = 0
    for fr in cv.fold_results:
        te = folds[(fr.repeat_idx, fr.fold_idx)]
        for d in cv.labels:
            assert np.array_equal(Y.iloc[te][d].values, np.asarray(fr.y_true[d])), \
                f"fold mismatch r{fr.repeat_idx} f{fr.fold_idx} {fr.method} {d}"
            n += 1
    return n


def pred_matrix(cv, method, drug, folds, n_iso):
    """(3, n_iso) out-of-fold probabilities: row = repeat, column = isolate position."""
    P = np.full((3, n_iso), np.nan)
    for fr in cv.fold_results:
        if fr.method != method:
            continue
        te = folds[(fr.repeat_idx, fr.fold_idx)]
        P[fr.repeat_idx, te] = np.asarray(fr.y_prob[drug], float)
    assert not np.isnan(P).any(), f"incomplete predictions for {method} {drug}"
    return P


def x5_isolate_inference(cvs, stats, Y, folds, idx_boot):
    n = len(Y)
    out = {}
    for model in MODELS:
        cv = cvs[model]
        out[model] = {}
        for d in DRUGS:
            y = Y[d].values.astype(int)
            best = stats[model]["drugs"][d]["MCC"]["best_chain"]
            P = {"BR": pred_matrix(cv, "binary_relevance", d, folds, n),
                 "Ens": pred_matrix(cv, "ensemble", d, folds, n),
                 "Best": pred_matrix(cv, best, d, folds, n)}
            y_pool = np.tile(y, 3)
            obs = {k: all_metrics(y_pool, P[k].ravel()) for k in P}
            # (a) paired isolate bootstrap
            boot = {k: {m: np.empty(N_BOOT) for m in METRICS4} for k in P}
            for b in range(N_BOOT):
                ii = idx_boot[b]
                yb = np.tile(y[ii], 3)
                for k in P:
                    mm = all_metrics(yb, P[k][:, ii].ravel())
                    for m in METRICS4:
                        boot[k][m][b] = mm[m]
            deltas = {}
            for comp, a in (("Ens_minus_BR", "Ens"), ("Best_minus_BR", "Best")):
                deltas[comp] = {}
                for m in METRICS4:
                    db = boot[a][m] - boot["BR"][m]
                    dobs = obs[a][m] - obs["BR"][m]
                    deltas[comp][m] = {
                        "delta_pooled": float(dobs),
                        "boot_mean": float(np.nanmean(db)),
                        "ci95": [float(np.nanpercentile(db, 2.5)), float(np.nanpercentile(db, 97.5))],
                        "boot_p": boot_pvalue(dobs, db),
                        "fold_mean_delta": float(np.mean(list(metric_by_fold(cv, "ensemble" if a == "Ens" else best, d, m, clinical=m in ("VME", "ME")).values()))
                                                 - np.mean(list(metric_by_fold(cv, "binary_relevance", d, m, clinical=m in ("VME", "ME")).values()))),
                    }
            # (b) per repeat (pooled over the 5 folds of each repeat)
            per_rep = {k: [all_metrics(y, P[k][r]) for r in range(3)] for k in P}
            per_rep_summary = {}
            for k in P:
                per_rep_summary[k] = {m: {"values": [per_rep[k][r][m] for r in range(3)],
                                          "mean": float(np.mean([per_rep[k][r][m] for r in range(3)])),
                                          "min": float(np.min([per_rep[k][r][m] for r in range(3)])),
                                          "max": float(np.max([per_rep[k][r][m] for r in range(3)]))}
                                      for m in METRICS4}
            per_rep_delta = {}
            for comp, a in (("Ens_minus_BR", "Ens"), ("Best_minus_BR", "Best")):
                per_rep_delta[comp] = {m: [per_rep[a][r][m] - per_rep["BR"][r][m] for r in range(3)] for m in ("AUROC", "MCC")}
            # (c) DeLong per repeat
            delong = {}
            for comp, a in (("Ens_minus_BR", "Ens"), ("Best_minus_BR", "Best")):
                tests = []
                for r in range(3):
                    t = delong_paired(y, P[a][r], P["BR"][r])
                    # sanity: DeLong AUCs equal sklearn AUCs
                    assert abs(t["auc_a"] - roc_auc_score(y, P[a][r])) < 1e-9
                    assert abs(t["auc_b"] - roc_auc_score(y, P["BR"][r])) < 1e-9
                    t["repeat"] = r
                    tests.append(t)
                ps = [t["p"] for t in tests]
                delong[comp] = {"per_repeat": tests,
                                "n_repeats_positive": int(sum(t["diff"] > 0 for t in tests)),
                                "n_repeats_p_below_0.05": int(sum(p < 0.05 for p in ps)),
                                "min_p": float(min(ps)), "max_p": float(max(ps)),
                                "note": "one paired test per repeat on the same 187 isolates; the three repeats share isolates and are not independent, so they are not combined into a single p-value"}
            # (d) MDE from the per-fold and per-repeat differences (Ens - BR)
            mde = {}
            for m in ("AUROC", "MCC"):
                e, b_ = aligned(metric_by_fold(cv, "ensemble", d, m), metric_by_fold(cv, "binary_relevance", d, m))
                diffs = e - b_
                keys = sorted(metric_by_fold(cv, "ensemble", d, m))
                rep_means = [float(np.mean([diffs[i] for i, k in enumerate(keys) if k[0] == r])) for r in range(3)]
                sd15 = float(np.std(diffs, ddof=1))
                sd3 = float(np.std(rep_means, ddof=1))
                mde[m] = {"observed_delta_folds": float(diffs.mean()), "sd_fold_diffs": sd15,
                          "mde_15_folds": mde_paired(sd15, 15),
                          "per_repeat_mean_diffs": rep_means, "sd_repeat_diffs": sd3,
                          "mde_3_repeats": mde_paired(sd3, 3)}
            out[model][d] = {"best_chain_MCC": best, "pooled_observed": obs, "bootstrap": deltas,
                             "per_repeat": per_rep_summary, "per_repeat_delta": per_rep_delta,
                             "delong": delong, "mde": mde}
            print(f"  {model} {d}: dAUROC(Ens-BR) pooled {deltas['Ens_minus_BR']['AUROC']['delta_pooled']:+.3f} "
                  f"CI {deltas['Ens_minus_BR']['AUROC']['ci95']} p={deltas['Ens_minus_BR']['AUROC']['boot_p']:.3f}; "
                  f"dMCC {deltas['Ens_minus_BR']['MCC']['delta_pooled']:+.3f} p={deltas['Ens_minus_BR']['MCC']['boot_p']:.3f}; "
                  f"DeLong p per repeat {[round(t['p'], 3) for t in delong['Ens_minus_BR']['per_repeat']]}")
    return out


def x8_honest_best(cvs, stats):
    out = {}
    for model in MODELS:
        cv = cvs[model]
        out[model] = {}
        for d in DRUGS:
            out[model][d] = {}
            for met in ("AUROC", "MCC"):
                br = metric_by_fold(cv, "binary_relevance", d, met)
                fm = {o: metric_by_fold(cv, o, d, met) for o in ORDERINGS}
                keys = sorted(br)
                rot = []
                for r_out in range(3):
                    sel = [k for k in keys if k[0] != r_out]
                    ev = [k for k in keys if k[0] == r_out]
                    means = {o: float(np.mean([fm[o][k] for k in sel])) for o in ORDERINGS}
                    o_sel = max(ORDERINGS, key=lambda o: means[o])
                    gain = float(np.mean([fm[o_sel][k] - br[k] for k in ev]))
                    rot.append({"held_out_repeat": r_out, "selected": o_sel, "selection_mean_10_folds": means[o_sel],
                                "gain_on_held_out_5_folds": gain})
                honest_rot = float(np.mean([x["gain_on_held_out_5_folds"] for x in rot]))
                lofo = []
                for k in keys:
                    others = [kk for kk in keys if kk != k]
                    means = {o: float(np.mean([fm[o][kk] for kk in others])) for o in ORDERINGS}
                    o_sel = max(ORDERINGS, key=lambda o: means[o])
                    lofo.append({"fold": list(k), "selected": o_sel, "gain": float(fm[o_sel][k] - br[k])})
                g = np.array([x["gain"] for x in lofo])
                _, p_lofo = exact_sign_flip_pvalue(g)
                _, lo, hi = bca_mean_ci(g)
                s = stats[model]["drugs"][d][met]
                out[model][d][met] = {
                    "BR_mean": float(np.mean(list(br.values()))),
                    "in_sample_best": {"ordering": s["best_chain"], "delta": s["delta_Best"]["delta"],
                                       "ci95": s["delta_Best"]["ci95"], "perm_p": s["delta_Best"]["perm_p"]},
                    "ensemble": {"delta": s["delta_Ens"]["delta"], "ci95": s["delta_Ens"]["ci95"], "perm_p": s["delta_Ens"]["perm_p"]},
                    "honest_repeat_rotation": {"gain": honest_rot, "rotations": rot},
                    "honest_leave_one_fold_out": {"gain": float(g.mean()), "ci95": [lo, hi], "perm_p": p_lofo,
                                                  "selected_counts": {short(o): int(sum(x["selected"] == o for x in lofo)) for o in ORDERINGS if any(x["selected"] == o for x in lofo)}},
                }
                print(f"  {model} {d} {met}: in-sample best {s['delta_Best']['delta']:+.3f} [{short(s['best_chain'])}] | "
                      f"honest rotation {honest_rot:+.3f} ({', '.join(short(x['selected']) for x in rot)}) | LOFO {g.mean():+.3f} p={p_lofo:.3f} | ensemble {s['delta_Ens']['delta']:+.3f}")
    return out


def x9a_multiplicity(stats):
    from statsmodels.stats.multitest import multipletests
    rows = []
    for model in MODELS:
        for d in DRUGS:
            for met in ("AUROC", "MCC"):
                s = stats[model]["drugs"][d][met]
                rows.append({"model": model, "drug": d, "metric": met, "delta_Ens": s["delta_Ens"]["delta"],
                             "ci95": s["delta_Ens"]["ci95"], "perm_p": s["delta_Ens"]["perm_p"],
                             "primary": (model, d) == PRIMARY})
    p = np.array([r["perm_p"] for r in rows])
    holm = multipletests(p, method="holm")[1]
    bh = multipletests(p, method="fdr_bh")[1]
    assert np.allclose(holm, _sig.holm_adjust(p))
    for r, h, b in zip(rows, holm, bh):
        r["p_holm_18"] = float(h)
        r["p_bh_18"] = float(b)
    sec = [i for i, r in enumerate(rows) if not r["primary"]]
    p16 = p[sec]
    holm16 = multipletests(p16, method="holm")[1]
    bh16 = multipletests(p16, method="fdr_bh")[1]
    for i, h, b in zip(sec, holm16, bh16):
        rows[i]["p_holm_16"] = float(h)
        rows[i]["p_bh_16"] = float(b)
    for r in rows:
        r.setdefault("p_holm_16", None)
        r.setdefault("p_bh_16", None)
    return {"n_tests": len(rows), "tests": rows,
            "survive_0.05": {"raw": [f"{r['model']} {r['drug']} {r['metric']}" for r in rows if r["perm_p"] < 0.05],
                             "holm_18": [f"{r['model']} {r['drug']} {r['metric']}" for r in rows if r["p_holm_18"] < 0.05],
                             "bh_18": [f"{r['model']} {r['drug']} {r['metric']}" for r in rows if r["p_bh_18"] < 0.05],
                             "holm_16_without_primary": [f"{r['model']} {r['drug']} {r['metric']}" for r in rows if r["p_holm_16"] is not None and r["p_holm_16"] < 0.05],
                             "bh_16_without_primary": [f"{r['model']} {r['drug']} {r['metric']}" for r in rows if r["p_bh_16"] is not None and r["p_bh_16"] < 0.05]}}


def load_cooc():
    out = {}
    for s in SITES:
        j = json.loads((ROOT / f"results/cooccurrence/{s}/cooccurrence_results.json").read_text())
        pm = j["phi_matrix"]
        rr = j["resistance_rates"]
        ceph = "CTX" if "CTX" in pm else "CRO"
        canon = {"CIP": "CIP", ceph: "CEPH", "CAZ": "CAZ"}
        out[s] = {"n": int(j["n_samples"]), "ceph_key": ceph,
                  "phi": {f"{canon[a]}-{canon[b]}": float(pm[a][b]) for a in canon for b in canon if a != b},
                  "phi_max": {canon[a]: float(max(pm[a][b] for b in pm[a] if b != a)) for a in canon},
                  "prev": {canon[a]: float(rr[a]) for a in canon}}
    return out


def cross_frame(cooc):
    cd = json.loads((ROOT / "results/cross_dataset/logistic/cross_dataset_results.json").read_text())
    rows = []
    for e in cd:
        tr, te = SITE_FROM_CROSS[e["train"]], SITE_FROM_CROSS[e["test"]]
        br, en = e["methods"]["binary_relevance"], e["methods"]["ensemble"]
        for dk in br:
            canon = "CEPH" if dk in ("CRO", "CTX") else dk
            rows.append(dict(
                train=tr, test=te, pair=f"{tr}->{te}", drug=canon, drug_key=dk,
                n_train=int(e["n_train"]), n_test=int(e["n_test"]), log_n_train=float(np.log10(e["n_train"])),
                auroc_br=br[dk]["AUROC"], auroc_ens=en[dk]["AUROC"], gain_auroc=en[dk]["AUROC"] - br[dk]["AUROC"],
                mcc_br=br[dk]["MCC"], mcc_ens=en[dk]["MCC"], gain_mcc=en[dk]["MCC"] - br[dk]["MCC"],
                phi_tr_cip_ceph=cooc[tr]["phi"]["CIP-CEPH"], phi_tr_ceph_caz=cooc[tr]["phi"]["CEPH-CAZ"],
                phi_mismatch=abs(cooc[tr]["phi_max"][canon] - cooc[te]["phi_max"][canon]),
                prev_train=cooc[tr]["prev"][canon], prev_test=cooc[te]["prev"][canon],
                prev_shift=cooc[tr]["prev"][canon] - cooc[te]["prev"][canon]))
    df = pd.DataFrame(rows)
    assert len(df) == 60 and df["pair"].nunique() == 20
    return df


def fit_lmm(df, ycol, covs, group, drug_fe):
    import statsmodels.formula.api as smf
    rhs = " + ".join(covs) + (" + C(drug)" if drug_fe else "")
    # Powell first: with a train-site random intercept the train-site-constant phi covariates
    # have no within-group variation, and L-BFGS can push the random-effect variance to a value
    # where X'V^-1X is numerically singular (statsmodels raises LinAlgError). Powell, Nelder-Mead
    # and L-BFGS agree to three decimals wherever all three converge.
    res, used = None, None
    for meth in ("powell", "nm", "lbfgs"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md = smf.mixedlm(f"{ycol} ~ {rhs}", df, groups=df[group])
                res = md.fit(reml=True, method=meth, maxiter=5000)
            used = meth
            if res.converged:
                break
        except np.linalg.LinAlgError:
            continue
    assert res is not None, f"no optimizer succeeded for {ycol} ~ {rhs} | {group}"
    out = {"formula": f"{ycol} ~ {rhs}", "n_obs": int(len(df)), "random_intercept": group,
           "n_groups": int(df[group].nunique()), "converged": bool(res.converged), "optimizer": used,
           "re_var": float(np.asarray(res.cov_re)[0, 0]), "resid_var": float(res.scale), "terms": {}}
    for c in covs:
        b = float(res.fe_params[c])
        se = float(res.bse[c])
        out["terms"][c] = {"slope": b, "se": se, "ci95": [b - 1.96 * se, b + 1.96 * se],
                           "p": float(res.pvalues[c]), "cov_sd": float(df[c].std(ddof=1)),
                           "slope_per_sd": b * float(df[c].std(ddof=1))}
    return out


def x9b_cross_site(cooc):
    df = cross_frame(cooc)
    sign = {}
    for d in ("CIP", "CEPH", "CAZ"):
        sub = df[df.drug == d].sort_values("pair")
        sign[d] = {}
        for met, col in (("AUROC", "gain_auroc"), ("MCC", "gain_mcc")):
            g = sub[col].values
            obs, p = exact_sign_flip_chunked(g)
            m, lo, hi = bca_mean_ci(g)
            sign[d][met] = {"n_pairs": int(len(g)), "mean_gain": obs, "bca_ci95": [lo, hi], "sign_flip_p": p,
                            "n_positive": int((g > 0).sum()), "mean_BR": float(sub[col.replace("gain_", "") + "_br"].mean()),
                            "mean_Ens": float(sub[col.replace("gain_", "") + "_ens"].mean())}
            print(f"  sign-flip {d} {met}: mean gain {obs:+.4f} BCa [{lo:+.4f}, {hi:+.4f}] p={p:.3f} ({int((g > 0).sum())}/20 positive)")
    mixed = {}
    for ycol, name in (("auroc_ens", "transfer_AUROC_ensemble"), ("gain_auroc", "chain_gain_AUROC")):
        mixed[name] = {}
        for d in ("CIP", "CEPH", "CAZ", "pooled"):
            sub = df if d == "pooled" else df[df.drug == d]
            block = {"multivariable_test_site_RE": fit_lmm(sub, ycol, COVS, "test", d == "pooled"),
                     "multivariable_train_site_RE": fit_lmm(sub, ycol, COVS, "train", d == "pooled"),
                     "univariable_test_site_RE": {c: fit_lmm(sub, ycol, [c], "test", d == "pooled") for c in COVS},
                     # the phi(train) covariates are train-site constants and are confounded with the
                     # training-set size (DRIAMS-A: lowest ceph-CAZ phi, largest n); log10 n_train is added as a sensitivity
                     "multivariable_test_site_RE_plus_log_n_train": fit_lmm(sub, ycol, COVS + ["log_n_train"], "test", d == "pooled")}
            mixed[name][d] = block
            t = block["multivariable_test_site_RE"]
            print(f"  LMM {name} {d} (test-site RE, converged={t['converged']}): " +
                  "; ".join(f"{c} {t['terms'][c]['slope']:+.3f} p={t['terms'][c]['p']:.3f}" for c in COVS))
    cov_summary = {c: {"min": float(df[c].min()), "max": float(df[c].max()), "sd": float(df[c].std(ddof=1)),
                       "n_distinct": int(df[c].round(6).nunique())} for c in COVS + ["log_n_train"]}
    by_train = df.groupby("train").agg(n_train=("n_train", "first"), phi_ceph_caz=("phi_tr_ceph_caz", "first"),
                                       mean_ens_auroc=("auroc_ens", "mean"), mean_gain_auroc=("gain_auroc", "mean"))
    cov_summary["by_train_site"] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in by_train.to_dict(orient="index").items()}
    return {"frame": df.to_dict(orient="records"), "sign_flip": sign, "mixed_models": mixed,
            "covariate_summary": cov_summary,
            "site_n": {s: cooc[s]["n"] for s in SITES},
            "notes": ["pairs sharing a train or test site are not independent; the sign-flip test treats the 20 pairs as exchangeable",
                      "phi_mismatch = |phi_max(train, drug) - phi_max(test, drug)| where phi_max is the largest phi between the drug and the other two drugs at that site",
                      "prev_shift = resistance prevalence(train) - prevalence(test) for the drug",
                      "phi_tr_* covariates are train-site constants (5 distinct values) and are confounded with training-set size and site identity; CEPH = CTX at Sant'Orsola and CRO at the DRIAMS sites",
                      "DRIAMS-B n = 213 in the cooccurrence file (the manuscript's 212 is wrong)"]}


def x9c_training_scheme(cv_can, cv_sens):
    rows = []
    for d in DRUGS:
        for met in METRICS4:
            clin = met in ("VME", "ME")
            row = {"drug": d, "metric": met}
            for name, cv in (("canonical", cv_can), ("cvsens", cv_sens)):
                br = metric_by_fold(cv, "binary_relevance", d, met, clinical=clin)
                ens = metric_by_fold(cv, "ensemble", d, met, clinical=clin)
                best = best_chain(cv, d, "MCC" if clin else met)
                bc = metric_by_fold(cv, best, d, met, clinical=clin)
                e, b_ = aligned(ens, br)
                c, b2 = aligned(bc, br)
                _, pE = exact_sign_flip_pvalue(e - b_)
                _, pB = exact_sign_flip_pvalue(c - b2)
                rec = {"BR": float(np.mean(list(br.values()))), "Ens": float(np.mean(list(ens.values()))),
                       "Best": float(np.mean(list(bc.values()))), "best_chain": best,
                       "delta_Ens": float((e - b_).mean()), "p_Ens": pE,
                       "delta_Best": float((c - b2).mean()), "p_Best_raw": pB}
                if not clin:
                    D = np.stack([aligned(metric_by_fold(cv, o, d, met), br)[0] - aligned(metric_by_fold(cv, o, d, met), br)[1] for o in ORDERINGS])
                    _, _, p_max, _ = best_chain_max_stat_pvalue(D)
                    rec["p_Best_maxstat"] = p_max
                row[name] = rec
            rows.append(row)
    # BR identical across schemes?
    diffs = []
    for fr in cv_can.fold_results:
        if fr.method != "binary_relevance":
            continue
        other = [g for g in cv_sens.fold_results if g.method == "binary_relevance" and g.repeat_idx == fr.repeat_idx and g.fold_idx == fr.fold_idx][0]
        for d in DRUGS:
            diffs.append(float(np.max(np.abs(np.asarray(fr.y_prob[d]) - np.asarray(other.y_prob[d])))))
    return {"rows": rows, "max_abs_diff_BR_probabilities": max(diffs)}


def write_tables(res):
    L = ["# Inference checks: Sant'Orsola nested CV and cross-site transfer", ""]
    L += ["##. Paired isolate bootstrap of the pooled out-of-fold predictions (2000 resamples of the 187 isolates; each isolate contributes its 3 out-of-fold predictions)", "",
          "Delta = chain minus BR on the pooled resampled predictions; percentile 95 percent interval; two-sided bootstrap p. Best = MCC-best ordering (in-sample). fold-mean delta = manuscript estimate (mean of 15 per-fold differences).", ""]
    hdr = ["Model", "Drug", "Comparison", "Metric", "pooled delta", "fold-mean delta", "95% CI", "boot p"]
    rows = []
    for model in MODELS:
        for d in DRUGS:
            x = res["X5"][model][d]
            for comp, lab in (("Ens_minus_BR", "Ens - BR"), ("Best_minus_BR", f"Best - BR [{short(x['best_chain_MCC'])}]")):
                for m in METRICS4:
                    b = x["bootstrap"][comp][m]
                    rows.append([model, d, lab, m, f"{b['delta_pooled']:+.3f}", f"{b['fold_mean_delta']:+.3f}",
                                 f"[{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}]", fp(b["boot_p"])])
    L += [md_table(hdr, rows), ""]
    L += ["##. Per-repeat estimates (metric on the pooled predictions of the 187 isolates within each repeat)", ""]
    hdr = ["Model", "Drug", "Metric", "BR r0/r1/r2", "BR mean [range]", "Ens r0/r1/r2", "Ens mean [range]", "Ens - BR r0/r1/r2", "Best - BR r0/r1/r2"]
    rows = []
    for model in MODELS:
        for d in DRUGS:
            x = res["X5"][model][d]
            for m in ("AUROC", "MCC"):
                br = x["per_repeat"]["BR"][m]
                en = x["per_repeat"]["Ens"][m]
                de = x["per_repeat_delta"]["Ens_minus_BR"][m]
                dbst = x["per_repeat_delta"]["Best_minus_BR"][m]
                rows.append([model, d, m, "/".join(f"{v:.3f}" for v in br["values"]), f"{br['mean']:.3f} [{br['min']:.3f}, {br['max']:.3f}]",
                             "/".join(f"{v:.3f}" for v in en["values"]), f"{en['mean']:.3f} [{en['min']:.3f}, {en['max']:.3f}]",
                             "/".join(f"{v:+.3f}" for v in de), "/".join(f"{v:+.3f}" for v in dbst)])
    L += [md_table(hdr, rows), ""]
    L += ["##. Paired DeLong tests per repeat (same 187 isolates; repeats are not independent and are not combined)", ""]
    hdr = ["Model", "Drug", "Comparison", "AUROC diff r0/r1/r2", "DeLong p r0/r1/r2", "repeats with p < 0.05"]
    rows = []
    for model in MODELS:
        for d in DRUGS:
            x = res["X5"][model][d]
            for comp, lab in (("Ens_minus_BR", "Ens - BR"), ("Best_minus_BR", "Best - BR")):
                t = x["delong"][comp]
                rows.append([model, d, lab, "/".join(f"{u['diff']:+.3f}" for u in t["per_repeat"]),
                             "/".join(fp(u["p"]) for u in t["per_repeat"]), f"{t['n_repeats_p_below_0.05']}/3"])
    L += [md_table(hdr, rows), ""]
    L += ["## X5d. Minimum detectable paired difference (Ens - BR) at 80 percent power, alpha 0.05 two-sided", "",
          "Paired t-test approximation using the observed SD of the per-fold differences (n = 15 folds; folds are not independent, so the MDE is optimistic) and of the per-repeat mean differences (n = 3).", ""]
    hdr = ["Model", "Drug", "Metric", "observed delta", "SD (15 folds)", "MDE (n = 15)", "SD (3 repeats)", "MDE (n = 3)"]
    rows = []
    for model in MODELS:
        for d in DRUGS:
            for m in ("AUROC", "MCC"):
                z = res["X5"][model][d]["mde"][m]
                rows.append([model, d, m, f"{z['observed_delta_folds']:+.3f}", f"{z['sd_fold_diffs']:.3f}", f"{z['mde_15_folds']:.3f}",
                             f"{z['sd_repeat_diffs']:.3f}", f"{z['mde_3_repeats']:.3f}"])
    L += [md_table(hdr, rows), ""]
    L += ["##. Honest best-ordering gain (ordering selected on the 10 folds of the other two repeats, evaluated on the 5 folds of the held-out repeat; mean of the three rotations) next to the in-sample best-of-six and the ensemble", "",
          "LOFO = leave-one-fold-out selection (select on 14 folds, evaluate on the held-out fold; 15 gains; exact sign-flip p).", ""]
    hdr = ["Model", "Drug", "Metric", "BR", "in-sample best (ordering)", "perm p", "honest rotation gain", "selected per rotation (r0, r1, r2)", "honest LOFO gain [BCa 95%]", "LOFO p", "ensemble gain", "perm p"]
    rows = []
    for model in MODELS:
        for d in ("CTX", "CAZ", "CIP"):
            for m in ("AUROC", "MCC"):
                z = res["X8"][model][d][m]
                rows.append([model, d, m, f"{z['BR_mean']:.3f}",
                             f"{z['in_sample_best']['delta']:+.3f} ({short(z['in_sample_best']['ordering'])})", fp(z["in_sample_best"]["perm_p"]),
                             f"{z['honest_repeat_rotation']['gain']:+.3f}",
                             ", ".join(short(r["selected"]) for r in z["honest_repeat_rotation"]["rotations"]),
                             f"{z['honest_leave_one_fold_out']['gain']:+.3f} [{z['honest_leave_one_fold_out']['ci95'][0]:+.3f}, {z['honest_leave_one_fold_out']['ci95'][1]:+.3f}]",
                             fp(z["honest_leave_one_fold_out"]["perm_p"]),
                             f"{z['ensemble']['delta']:+.3f}", fp(z["ensemble"]["perm_p"])])
    L += [md_table(hdr, rows), ""]
    L += ["##. Multiplicity over the 18 ensemble-versus-BR tests (exact sign-flip p from within_site_stats.json)", "",
          "Holm and Benjamini-Hochberg adjusted within the family of 18; the 16-test columns exclude the logistic regression CTX comparisons (AUROC and MCC), as a sensitivity check.", ""]
    hdr = ["Model", "Drug", "Metric", "delta Ens", "raw p", "Holm (18)", "BH (18)", "Holm (16, without logistic CTX)", "BH (16, without logistic CTX)"]
    rows = []
    for r in res["X9a"]["tests"]:
        tag = " (logistic CTX)" if r["primary"] else ""
        rows.append([r["model"], r["drug"] + tag, r["metric"], f"{r['delta_Ens']:+.3f}", fp(r["perm_p"]), fp(r["p_holm_18"]), fp(r["p_bh_18"]),
                     "n/a" if r["p_holm_16"] is None else fp(r["p_holm_16"]), "n/a" if r["p_bh_16"] is None else fp(r["p_bh_16"])])
    L += [md_table(hdr, rows), ""]
    s = res["X9a"]["survive_0.05"]
    L += ["Survive at 0.05: raw " + str(len(s["raw"])) + " (" + "; ".join(s["raw"]) + "); Holm (18): " + ("; ".join(s["holm_18"]) or "none") +
          "; BH (18): " + ("; ".join(s["bh_18"]) or "none") + "; Holm (16): " + ("; ".join(s["holm_16_without_primary"]) or "none") +
          "; BH (16): " + ("; ".join(s["bh_16_without_primary"]) or "none") + ".", ""]
    L += ["##. Cross-site chain gain (ensemble - BR) over the 20 train-test pairs (logistic regression)", "",
          "Exact paired sign-flip test over the 20 pairs and BCa 95 percent interval of the mean. Pairs sharing a train or test site are not independent.", ""]
    hdr = ["Drug", "Metric", "mean BR", "mean Ens", "mean gain", "BCa 95% CI", "positive pairs", "sign-flip p"]
    rows = []
    for d, lab in (("CIP", "CIP"), ("CEPH", "CRO/CTX"), ("CAZ", "CAZ")):
        for m in ("AUROC", "MCC"):
            z = res["X9b"]["sign_flip"][d][m]
            rows.append([lab, m, f"{z['mean_BR']:.3f}", f"{z['mean_Ens']:.3f}", f"{z['mean_gain']:+.4f}", f"[{z['bca_ci95'][0]:+.4f}, {z['bca_ci95'][1]:+.4f}]", f"{z['n_positive']}/20", fp(z["sign_flip_p"])])
    L += [md_table(hdr, rows), ""]
    L += ["### Linear mixed models (random intercept per test site; sensitivity: per train site), multivariable with the four covariates", "",
          "Slopes are per unit of the covariate (phi and prevalence are proportions); 95 percent Wald intervals. phi(train) covariates are train-site constants (5 distinct values).", ""]
    hdr = ["Outcome", "Drug", "Covariate", "slope (test-site RE)", "95% CI", "p", "slope (train-site RE)", "p", "univariable slope (test RE)", "p"]
    rows = []
    for name, lab in (("transfer_AUROC_ensemble", "ensemble transfer AUROC"), ("chain_gain_AUROC", "chain gain (Ens - BR AUROC)")):
        for d, dl in (("CIP", "CIP"), ("CEPH", "CRO/CTX"), ("CAZ", "CAZ"), ("pooled", "pooled + drug FE")):
            blk = res["X9b"]["mixed_models"][name][d]
            a, b, u = blk["multivariable_test_site_RE"], blk["multivariable_train_site_RE"], blk["univariable_test_site_RE"]
            for c in COVS:
                ta, tb, tu = a["terms"][c], b["terms"][c], u[c]["terms"][c]
                rows.append([lab, dl, COV_LABEL[c], f"{ta['slope']:+.3f}", f"[{ta['ci95'][0]:+.3f}, {ta['ci95'][1]:+.3f}]", fp(ta["p"]),
                             f"{tb['slope']:+.3f}", fp(tb["p"]), f"{tu['slope']:+.3f}", fp(tu["p"])])
    L += [md_table(hdr, rows), ""]
    conv = []
    for name in ("transfer_AUROC_ensemble", "chain_gain_AUROC"):
        for d in ("CIP", "CEPH", "CAZ", "pooled"):
            blk = res["X9b"]["mixed_models"][name][d]
            conv.append(f"{name}/{d}: test-RE converged={blk['multivariable_test_site_RE']['converged']} (RE var {blk['multivariable_test_site_RE']['re_var']:.2e}), train-RE converged={blk['multivariable_train_site_RE']['converged']} (RE var {blk['multivariable_train_site_RE']['re_var']:.2e})")
    L += ["Convergence: " + "; ".join(conv) + ".", ""]
    bt = res["X9b"]["covariate_summary"]["by_train_site"]
    L += ["### Sensitivity: the four covariates plus log10 n_train (test-site random intercept)", "",
          "The phi(train) covariates are train-site constants and are confounded with the training-set size: " +
          "; ".join(f"{SITE_LABEL[s]} n = {int(bt[s]['n_train'])}, phi ceph-CAZ {bt[s]['phi_ceph_caz']:.3f}, mean ensemble transfer AUROC {bt[s]['mean_ens_auroc']:.3f}, mean chain gain {bt[s]['mean_gain_auroc']:+.4f}" for s in SITES) + ".", ""]
    hdr = ["Outcome", "Drug", "Covariate", "slope", "95% CI", "p"]
    rows = []
    for name, lab in (("transfer_AUROC_ensemble", "ensemble transfer AUROC"), ("chain_gain_AUROC", "chain gain (Ens - BR AUROC)")):
        for d, dl in (("CIP", "CIP"), ("CEPH", "CRO/CTX"), ("CAZ", "CAZ"), ("pooled", "pooled + drug FE")):
            t = res["X9b"]["mixed_models"][name][d]["multivariable_test_site_RE_plus_log_n_train"]
            for c in COVS + ["log_n_train"]:
                tt = t["terms"][c]
                rows.append([lab, dl, COV_LABEL[c], f"{tt['slope']:+.3f}", f"[{tt['ci95'][0]:+.3f}, {tt['ci95'][1]:+.3f}]", fp(tt["p"])])
    L += [md_table(hdr, rows), ""]
    L += ["##. Training scheme sensitivity (logistic regression): canonical (links trained on true upstream labels) versus cvsens (links trained on out-of-fold upstream predictions)", "",
          f"Means over the 15 outer folds; exact sign-flip p for Ens - BR and Best - BR (raw, and max-statistic selection-corrected for AUROC and MCC). Best = metric-best ordering within each scheme (MCC-best for VME and ME). BR is identical in the two runs (max absolute probability difference {res['X9c']['max_abs_diff_BR_probabilities']:.1e}).", ""]
    hdr = ["Drug", "Metric", "BR", "Ens canonical", "p", "Best canonical (ordering)", "p raw / max-stat", "Ens cvsens", "p", "Best cvsens (ordering)", "p raw / max-stat"]
    rows = []
    for r in res["X9c"]["rows"]:
        c, s_ = r["canonical"], r["cvsens"]
        pc = fp(c["p_Best_raw"]) + (" / " + fp(c["p_Best_maxstat"]) if "p_Best_maxstat" in c else "")
        ps = fp(s_["p_Best_raw"]) + (" / " + fp(s_["p_Best_maxstat"]) if "p_Best_maxstat" in s_ else "")
        rows.append([r["drug"], r["metric"], f"{c['BR']:.3f}", f"{c['Ens']:.3f} ({c['delta_Ens']:+.3f})", fp(c["p_Ens"]),
                     f"{c['Best']:.3f} ({c['delta_Best']:+.3f}; {short(c['best_chain'])})", pc,
                     f"{s_['Ens']:.3f} ({s_['delta_Ens']:+.3f})", fp(s_["p_Ens"]),
                     f"{s_['Best']:.3f} ({s_['delta_Best']:+.3f}; {short(s_['best_chain'])})", ps])
    L += [md_table(hdr, rows), ""]
    OUT_MD.parent.mkdir(exist_ok=True, parents=True)
    OUT_MD.write_text("\n".join(L) + "\n")
    return "\n".join(L)


def _grouped_axis(ax, ps, groups, group_names, inner_labels, name_y=-0.13):
    """x = one position per (group, inner) pair: inner labels as ticks, group names below, thin separators."""
    n_inner = len(inner_labels)
    xs = np.arange(len(groups) * n_inner)
    ax.set_xticks(xs)
    ax.set_xticklabels([inner_labels[i % n_inner] for i in xs])
    ax.set_xlim(-0.6, len(xs) - 0.4)
    for g, name in enumerate(group_names):
        ax.text(g * n_inner + (n_inner - 1) / 2.0, name_y, name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color=ps.INK)
        if g:
            ax.axvline(g * n_inner - 0.5, color=ps.GRID, lw=0.6, zorder=0)
    return xs


def _tight(ax, lows, highs, pad=0.08):
    v = np.asarray([x for x in list(lows) + list(highs) if x is not None and np.isfinite(x)], float)
    span = max(v.max() - v.min(), 1e-3)
    ax.set_ylim(v.min() - pad * span, v.max() + pad * span)


def make_figures(res):
    """Supplement figures from the stored results (inference_isolate_bootstrap, honest_best_ordering)."""
    from mecor.visualization import paper_style as ps
    ps.apply()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    learner_names = [ps.LEARNER_NAME[m] for m in MODELS]

    # Figure 1: isolate bootstrap deltas (chain minus BR on the pooled out-of-fold predictions)
    fig, axes = ps.figure("double", height=2.9, ncols=2)
    for i, (ax, met) in enumerate(zip(axes, ("AUROC", "MCC"))):
        xs = _grouped_axis(ax, ps, MODELS, learner_names, DRUGS)
        groups = [(m, d) for m in MODELS for d in DRUGS]
        lows, highs = [0.0], [0.0]
        for off, comp, color, lab in ((-0.18, "Ens_minus_BR", ps.ROLE["ensemble"], "Chain ensemble minus BR"),
                                      (0.18, "Best_minus_BR", ps.ROLE["best"], "Best chain minus BR")):
            b = [res["X5"][m][d]["bootstrap"][comp][met] for m, d in groups]
            y = np.array([v["delta_pooled"] for v in b])
            lo = np.array([v["ci95"][0] for v in b]); hi = np.array([v["ci95"][1] for v in b])
            ps.ci_point(ax, xs + off, y, lo, hi, color=color, label=lab)
            lows += list(lo); highs += list(hi)
        ps.hline(ax, 0.0)
        _tight(ax, lows, highs)
        ax.set_ylabel(f"Gain over binary relevance ({met})")
        ps.panel_label(ax, "ab"[i])
        ps.despine(ax, grid="y")
    fig.tight_layout(w_pad=2.0)
    ps.legend_above(fig, y=1.0)
    ps.save(fig, FIG_DIR / "inference_isolate_bootstrap")

    # Figure 2: honest best ordering (gain over BR, mean over the 15 folds)
    fig, axes = ps.figure("double", height=2.9, ncols=2)
    drugs2 = ["CTX", "CAZ"]
    for i, (ax, met) in enumerate(zip(axes, ("AUROC", "MCC"))):
        xs = _grouped_axis(ax, ps, MODELS, learner_names, drugs2)
        groups = [(m, d) for m in MODELS for d in drugs2]
        z = [res["X8"][m][d][met] for m, d in groups]
        lows, highs = [0.0], [0.0]
        series = [(-0.24, ps.ROLE["best"], False, "Best chain, selected in-sample",
                   [v["in_sample_best"]["delta"] for v in z],
                   [v["in_sample_best"]["ci95"][0] for v in z], [v["in_sample_best"]["ci95"][1] for v in z]),
                  (0.0, ps.ROLE["best"], True, "Best chain, selected on the other two repeats",
                   [v["honest_repeat_rotation"]["gain"] for v in z], None, None),
                  (0.24, ps.ROLE["ensemble"], False, "Chain ensemble (no selection)",
                   [v["ensemble"]["delta"] for v in z],
                   [v["ensemble"]["ci95"][0] for v in z], [v["ensemble"]["ci95"][1] for v in z])]
        for off, color, hollow, lab, y, lo, hi in series:
            ps.ci_point(ax, xs + off, y, lo, hi, color=color, hollow=hollow, label=lab)
            lows += list(lo) if lo is not None else list(y)
            highs += list(hi) if hi is not None else list(y)
        ps.hline(ax, 0.0)
        _tight(ax, lows, highs)
        ax.set_ylabel(f"Gain over binary relevance ({met})")
        ps.panel_label(ax, "ab"[i])
        ps.despine(ax, grid="y")
    fig.tight_layout(w_pad=2.0)
    ps.legend_above(fig, y=1.0)
    ps.save(fig, FIG_DIR / "honest_best_ordering")
    print(f"saved figures in {FIG_DIR}")


def plot_from_results():
    """Plot-only path: redraw the two supplement figures from results/stats/inference_checks.json."""
    make_figures(json.loads(OUT_JSON.read_text()))


def main():
    np.random.seed(SEED)
    cfg = Config.from_yaml(str(ROOT / "configs/s_orsola.yaml"))
    X, Y = load_dataset(cfg)
    meta = pd.read_csv(Config.from_yaml(str(ROOT / "configs/s_orsola.yaml")).metadata_path)
    assert X.index.isin(meta["ID"]).all() and len(X) == 187, "isolate ids do not match data/metadata.csv"
    folds = reconstruct_folds(X, Y)
    stats = json.loads(STATS_JSON.read_text())

    cvs = {m: load_cv_results(ROOT / f"results/s_orsola/{m}/cv_results.pkl") for m in MODELS}
    cv_sens = load_cv_results(ROOT / "results/cvsens/logistic/cv_results.pkl")
    verified = {}
    for name, cv in list(cvs.items()) + [("cvsens_logistic", cv_sens)]:
        verified[name] = verify_folds(cv, folds, Y)
        print(f"fold reconstruction verified for {name}: {verified[name]} (fold, method, drug) label arrays identical")

    rng = np.random.default_rng(SEED)
    idx_boot = rng.integers(0, len(Y), size=(N_BOOT, len(Y)))

    print("\n== isolate-level inference")
    x5 = x5_isolate_inference(cvs, stats, Y, folds, idx_boot)
    print("\n== honest best ordering")
    x8 = x8_honest_best(cvs, stats)
    print("\n== multiplicity")
    x9a = x9a_multiplicity(stats)
    for r in x9a["tests"]:
        print(f"  {r['model']:9s} {r['drug']} {r['metric']:5s} delta {r['delta_Ens']:+.3f} raw p {r['perm_p']:.4f} Holm18 {r['p_holm_18']:.4f} BH18 {r['p_bh_18']:.4f}"
              + ("" if r["p_holm_16"] is None else f" Holm16 {r['p_holm_16']:.4f} BH16 {r['p_bh_16']:.4f}") + (" PRIMARY" if r["primary"] else ""))
    print("  survive:", json.dumps(x9a["survive_0.05"]))
    print("\n== cross-site")
    cooc = load_cooc()
    x9b = x9b_cross_site(cooc)
    print("\n== canonical versus cvsens")
    x9c = x9c_training_scheme(cvs["logistic"], cv_sens)
    for r in x9c["rows"]:
        if r["metric"] in ("AUROC", "MCC"):
            c, s_ = r["canonical"], r["cvsens"]
            print(f"  {r['drug']} {r['metric']}: canonical BR {c['BR']:.3f} Ens {c['Ens']:.3f} (p {c['p_Ens']:.3f}) Best {c['Best']:.3f} | "
                  f"cvsens Ens {s_['Ens']:.3f} (p {s_['p_Ens']:.3f}) Best {s_['Best']:.3f}")

    res = {"settings": {"seed": SEED, "n_boot": N_BOOT, "n_isolates": int(len(Y)), "folds_verified": verified,
                        "primary_endpoint": "logistic regression, CTX, chain ensemble versus BR (AUROC and MCC)",
                        "best_definition_X5": "MCC-best ordering per drug and model from within_site_stats.json (in-sample)"},
           "X5": x5, "X8": x8, "X9a": x9a, "X9b": x9b, "X9c": x9c}
    OUT_JSON.parent.mkdir(exist_ok=True, parents=True)
    OUT_JSON.write_text(json.dumps(to_native(res), indent=1))
    write_tables(res)
    make_figures(res)
    print(f"\nsaved {OUT_JSON}\nsaved {OUT_MD}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inference checks (Sant'Orsola and cross-site)")
    ap.add_argument("--plot-only", action="store_true",
                    help="Only redraw the figures from the stored results/stats/inference_checks.json")
    if ap.parse_args().plot_only:
        plot_from_results()
    else:
        main()
