"""Clinical-decision bundle on the Sant'Orsola cohort, computed from the
saved out-of-fold predictions of the nested cross-validation (no refitting).

Question: does the chain ensemble add clinical value over binary relevance (BR) when the
tool is framed as decision support in the window between species identification and the
AST result? The comparators are BR, treat-all (empirical coverage of every isolate) and
treat-none; device thresholds (VME below 3 percent) are not applied as pass/fail.

Methods (per model: logistic regression primary, CatBoost and TabPFN secondary; per drug;
for BR, the chain ensemble (Ens) and the MCC-best ordering per drug (Best, taken from
results/stats/within_site_stats.json; when the drug is first in that ordering Best equals BR)):
  - decision-curve analysis: net benefit versus threshold probability (0.05 to 0.60, step
       0.01) from the pooled out-of-fold predictions (each isolate appears three times, once
       per repeat; ratios are prevalence-faithful), with a 1000-resample isolate bootstrap
       (isolates are resampled, so the three predictions of an isolate move together; the same
       resamples are used for every method, which makes the differences paired).
  - operating characteristics (sensitivity, specificity, PPV, NPV, LR+, LR-) with isolate
       bootstrap intervals at the 0.5 threshold, at the thresholds giving VME 0.10 and 0.15 on the
       pooled curve, and for a two-threshold rule-in / rule-out scheme (rule-in when PPV first
       reaches 0.80, rule-out when NPV first reaches 0.90, or the highest achievable value with at
       least 5 percent support); post-test probabilities at prevalences 0.10 to 0.42.
  - calibration: Brier score, prevalence-only Brier, Brier skill score, calibration intercept
       (slope fixed at 1) and slope (logistic recalibration on logit(p), p clipped to
       [0.001, 0.999]), expected calibration error (10 equal-count bins), log-loss, AUPRC, AUROC.
  - cross-fitted threshold: for each repeat the MCC-maximizing threshold is chosen on the pooled
       predictions of the other two repeats and applied to this repeat; MCC, VME and ME at that
       threshold and at 0.5, per repeat and per outer fold (exact paired sign-flip test).
  - correlated profile errors per outer fold (subset accuracy, Hamming loss, fraction with all
       three or at least two calls wrong, discordant fully-correct profiles) for Ens and every
       single ordering against BR, with the exact paired sign-flip test over the 15 folds.

Outputs: results/stats/clinical_decision.json, tables/clinical_decision.md,
         figures/supplementary/clinical_decision_{dca,calibration,vme_me}_<model>.{png,pdf}.
Deterministic (seed 42). Run from the repository root with the project interpreter.
`--plot-only` redraws the figures from the stored JSON (decision curves with their bootstrap bands, reliability curves,
VME/ME sweeps and cross-fitted thresholds are all stored) without touching the bootstrap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)

from mecor.config import Config  # noqa: E402
from mecor.cvio import load_cv_results  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402

SEED = 42
N_BOOT = 1000
MODELS = [m for m in ("logistic", "catboost", "tabpfn") if (ROOT / f"results/s_orsola/{m}/cv_results.pkl").exists()]
MODEL_NAMES = {"logistic": "Logistic regression", "catboost": "CatBoost", "tabpfn": "TabPFN"}
DRUGS = ["CIP", "CTX", "CAZ"]
DCA_THR = np.round(np.arange(0.05, 0.60 + 1e-9, 0.01), 2)
REPORT_THR = [0.10, 0.20, 0.30, 0.42, 0.50]
PREVALENCES = [0.10, 0.20, 0.30, 0.42]
VME_TARGETS = [0.10, 0.15]
PPV_TARGET, NPV_TARGET = 0.80, 0.90
MIN_SUPPORT = 0.05          # minimum fraction of pooled predictions behind a "highest achievable" PPV or NPV
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)   # 0.01 to 0.99
CLIP = 0.001
N_BINS = 10
STATS_JSON = ROOT / "results/stats/within_site_stats.json"
FIG_DIR = ROOT / "figures/supplementary"
STACK_DIR = ROOT / "results/nested_stacking/s_orsola_logistic_full"   # stacked out-of-fold predictions (scripts/04_stacking/01_nested_stacking.py, full grid)


def exact_sign_flip_pvalue(d):
    """Exact two-sided paired sign-flip test (copy of scripts/02_chain_vs_br/02_significance.py).

    Returns (observed mean, p)."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1)
    signs = 1 - 2 * bits
    null_means = (signs * d).mean(axis=1)
    obs = d.mean()
    p = float(np.mean(np.abs(null_means) >= np.abs(obs) - 1e-15))
    return float(obs), p


def ci(boot):
    boot = np.asarray(boot, float)
    return [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))]


def est(point, boot):
    return {"est": float(point), "ci": ci(boot)}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def f3(x):
    return "nan" if x is None or not np.isfinite(x) else f"{x:.3f}"


def fci(b):
    return f"{f3(b['est'])} [{f3(b['ci'][0])}, {f3(b['ci'][1])}]"


def fp(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def reconstruct_folds(X, Y):
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    return {(g // 5, g % 5): te for g, (tr, te) in enumerate(outer.split(X, strat))}


def pool(cv, folds, Y):
    """Return (iso, rep, fold, ytrue{drug}, prob{method}{drug}) with rows ordered by fold."""
    keys = sorted(folds)
    by = {(fr.method, fr.repeat_idx, fr.fold_idx): fr for fr in cv.fold_results}
    methods = sorted({fr.method for fr in cv.fold_results})
    iso = np.concatenate([folds[k] for k in keys])
    rep = np.concatenate([np.full(len(folds[k]), k[0]) for k in keys])
    fold = np.concatenate([np.full(len(folds[k]), k[1]) for k in keys])
    ytrue, prob = {}, {m: {} for m in methods}
    for d in cv.labels:
        ytrue[d] = np.concatenate([np.asarray(by[("binary_relevance",) + k].y_true[d]) for k in keys]).astype(float)
        assert np.array_equal(ytrue[d], Y.iloc[iso][d].values.astype(float)), "fold reconstruction mismatch"
        for m in methods:
            for k in keys:
                assert np.array_equal(np.asarray(by[(m,) + k].y_true[d]).astype(float),
                                      Y.iloc[folds[k]][d].values.astype(float)), "method alignment mismatch"
            prob[m][d] = np.concatenate([np.asarray(by[(m,) + k].y_prob[d]) for k in keys]).astype(float)
    return iso, rep, fold, ytrue, prob


def bootstrap_weights(n_iso, iso, rng):
    counts = rng.multinomial(n_iso, np.full(n_iso, 1.0 / n_iso), size=N_BOOT)
    return counts[:, iso].astype(float)   # N_BOOT x n_rows, each isolate's three rows share a weight


def confusion(y, p, thr, W=None):
    thr = np.atleast_1d(np.asarray(thr, float))
    pred = (p[:, None] >= thr[None, :]).astype(float)
    if W is None:
        W = np.ones((1, y.size))
    wy = W * y[None, :]
    wn = W * (1.0 - y)[None, :]
    TP = wy @ pred
    FP = wn @ pred
    P = wy.sum(1, keepdims=True)
    N = wn.sum(1, keepdims=True)
    return TP, FP, N - FP, P - TP


def characteristics(TP, FP, TN, FN):
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = TP / (TP + FN)
        spec = TN / (TN + FP)
        return {"sensitivity": sens, "specificity": spec, "PPV": TP / (TP + FP), "NPV": TN / (TN + FN),
                "LR_pos": sens / (1.0 - spec), "LR_neg": (1.0 - sens) / spec,
                "VME": 1.0 - sens, "ME": 1.0 - spec}


def mcc_from(TP, FP, TN, FN):
    with np.errstate(divide="ignore", invalid="ignore"):
        den = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        m = (TP * TN - FP * FN) / den
    return np.where(den > 0, m, 0.0)


def net_benefit(TP, FP, n, thr):
    return TP / n - FP / n * thr / (1.0 - thr)


def post_test(prev, lr):
    odds = prev / (1.0 - prev) * lr
    return odds / (1.0 + odds)


def decision_curve(y, p, W):
    TP, FP, TN, FN = confusion(y, p, DCA_THR)
    n = TP + FP + TN + FN
    nb = net_benefit(TP, FP, n, DCA_THR)[0]
    TPb, FPb, TNb, FNb = confusion(y, p, DCA_THR, W)
    nbb = net_benefit(TPb, FPb, TPb + FPb + TNb + FNb, DCA_THR)
    return nb, nbb


def treat_all_curve(y, W):
    prev = y.mean()
    nb = prev - (1 - prev) * DCA_THR / (1 - DCA_THR)
    prevb = (W * y).sum(1) / W.sum(1)
    nbb = prevb[:, None] - (1 - prevb)[:, None] * (DCA_THR / (1 - DCA_THR))[None, :]
    return nb, nbb


def ranges(mask):
    """Contiguous threshold ranges where a boolean mask over DCA_THR is true."""
    out, start, prev_t = [], None, None
    for t, m in zip(DCA_THR, mask):
        if m and start is None:
            start = t
        if not m and start is not None:
            out.append([float(start), float(prev_t)])
            start = None
        prev_t = t
    if start is not None:
        out.append([float(start), float(DCA_THR[-1])])
    return out


def vme_threshold(y, p, target):
    """Highest threshold whose pooled sensitivity is at least 1 - target."""
    pos = np.sort(p[y == 1])[::-1]
    k = int(np.ceil((1 - target) * pos.size))
    return float(pos[k - 1])


def rule_thresholds(y, p):
    TP, FP, TN, FN = confusion(y, p, GRID)
    ch = characteristics(TP, FP, TN, FN)
    n = float(y.size)
    ppv, npv = ch["PPV"][0], ch["NPV"][0]
    pos_frac, neg_frac = (TP + FP)[0] / n, (TN + FN)[0] / n
    ok_in = (ppv >= PPV_TARGET) & (pos_frac >= MIN_SUPPORT)
    if ok_in.any():
        t_in, in_met = float(GRID[ok_in].min()), True
    else:
        sup = pos_frac >= MIN_SUPPORT
        cand = np.where(sup, ppv, -np.inf)
        t_in, in_met = float(GRID[int(np.flatnonzero(cand == cand.max())[0])]), False
    ok_out = (npv >= NPV_TARGET) & (neg_frac >= MIN_SUPPORT)
    if ok_out.any():
        t_out, out_met = float(GRID[ok_out].max()), True
    else:
        sup = neg_frac >= MIN_SUPPORT
        cand = np.where(sup, npv, -np.inf)
        t_out, out_met = float(GRID[int(np.flatnonzero(cand == cand.max())[-1])]), False
    overlap = t_out > t_in
    if overlap:
        t_out = t_in
    return {"thr_in": t_in, "ppv_target_met": in_met, "thr_out": t_out, "npv_target_met": out_met,
            "overlap_collapsed_to_single_threshold": bool(overlap)}


def scheme_stats(y, p, t_in, t_out, W):
    call_r = (p >= t_in).astype(float)
    call_s = (p < t_out).astype(float)
    act = call_r + call_s
    n = W.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "frac_rule_in": (W * call_r).sum(1) / n,
            "frac_rule_out": (W * call_s).sum(1) / n,
            "frac_actionable": (W * act).sum(1) / n,
            "PPV_rule_in": (W * call_r * y).sum(1) / (W * call_r).sum(1),
            "NPV_rule_out": (W * call_s * (1 - y)).sum(1) / (W * call_s).sum(1),
            "coverage_resistant": (W * act * y).sum(1) / (W * y).sum(1),
            "coverage_susceptible": (W * act * (1 - y)).sum(1) / (W * (1 - y)).sum(1),
            "sensitivity_among_actionable": (W * call_r * y).sum(1) / (W * act * y).sum(1),
            "specificity_among_actionable": (W * call_s * (1 - y)).sum(1) / (W * act * (1 - y)).sum(1),
        }
    return out


def operating_block(y, p, W, thr):
    TP, FP, TN, FN = confusion(y, p, thr)
    ch = characteristics(TP, FP, TN, FN)
    TPb, FPb, TNb, FNb = confusion(y, p, thr, W)
    chb = characteristics(TPb, FPb, TNb, FNb)
    out = {"threshold": float(thr), "n_pred_resistant_frac": float((TP + FP)[0, 0] / y.size)}
    for k in ("sensitivity", "specificity", "PPV", "NPV", "LR_pos", "LR_neg", "VME", "ME"):
        out[k] = est(ch[k][0, 0], chb[k][:, 0])
    out["post_test"] = {}
    for prev in PREVALENCES:
        out["post_test"][f"{prev:.2f}"] = {
            "after_resistant_call": est(post_test(prev, ch["LR_pos"][0, 0]), post_test(prev, chb["LR_pos"][:, 0])),
            "after_susceptible_call": est(post_test(prev, ch["LR_neg"][0, 0]), post_test(prev, chb["LR_neg"][:, 0])),
        }
    out["_boot"] = {k: chb[k][:, 0] for k in ("sensitivity", "specificity", "PPV", "NPV", "LR_pos", "LR_neg")}
    return out


def logit(p):
    pc = np.clip(p, CLIP, 1 - CLIP)
    return np.log(pc / (1 - pc))


def _loglik(eta, y, W):
    return (W * (y[None, :] * eta - np.logaddexp(0.0, eta))).sum(1)


def _damped(a, b, da, db, x, y, W, ll):
    """Backtracking line search per row so that the weighted log-likelihood never decreases."""
    step = np.ones(a.size)
    for _ in range(40):
        a_new, b_new = a + step * da, b + step * db
        ll_new = _loglik(a_new[:, None] + b_new[:, None] * x[None, :], y, W)
        bad = ~(ll_new >= ll - 1e-12)
        if not bad.any():
            break
        step[bad] *= 0.5
    return a_new, b_new, ll_new, step


def wlogit_slope(x, y, W, n_iter=200):
    """Weighted logistic fit of y on a + b x (damped Newton-Raphson, vectorized over rows of W)."""
    from scipy.special import expit
    B = W.shape[0]
    a, b = np.zeros(B), np.ones(B)
    ll = _loglik(a[:, None] + b[:, None] * x[None, :], y, W)
    for _ in range(n_iter):
        eta = a[:, None] + b[:, None] * x[None, :]
        mu = expit(eta)
        r = W * (y[None, :] - mu)
        h = W * mu * (1 - mu)
        ga, gb = r.sum(1), (r * x).sum(1)
        Haa, Hab, Hbb = h.sum(1), (h * x).sum(1), (h * x * x).sum(1)
        det = Haa * Hbb - Hab ** 2
        da = (Hbb * ga - Hab * gb) / det
        db = (Haa * gb - Hab * ga) / det
        a, b, ll, step = _damped(a, b, da, db, x, y, W, ll)
        if max(np.abs(step * da).max(), np.abs(step * db).max()) < 1e-9:
            break
    return a, b


def wlogit_intercept(x, y, W, n_iter=200):
    """Calibration-in-the-large: intercept with the slope fixed at 1 (offset model, damped Newton)."""
    from scipy.special import expit
    a = np.zeros(W.shape[0])
    ones = np.ones_like(a)
    ll = _loglik(a[:, None] + x[None, :], y, W)
    for _ in range(n_iter):
        mu = expit(a[:, None] + x[None, :])
        da = (W * (y[None, :] - mu)).sum(1) / (W * mu * (1 - mu)).sum(1)
        a, _, ll, step = _damped(a, ones, da, np.zeros_like(a), x, y, W, ll)
        if np.abs(step * da).max() < 1e-9:
            break
    return a


def equal_count_bins(p, n_bins=N_BINS):
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    return np.searchsorted(edges[1:-1], p, side="right")


def binned(y, p, bins, W, n_bins=N_BINS):
    """Weighted per-bin mean prediction, observed frequency and weight share (rows of W)."""
    n = W.sum(1)
    pb, yb, share = [], [], []
    for b in range(n_bins):
        m = (bins == b).astype(float)
        wb = (W * m).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pb.append((W * m * p).sum(1) / wb)
            yb.append((W * m * y).sum(1) / wb)
        share.append(wb / n)
    return np.array(pb).T, np.array(yb).T, np.array(share).T   # B x n_bins


def calibration_block(y, p, W):
    ones = np.ones((1, y.size))
    x = logit(p)
    pc = np.clip(p, CLIP, 1 - CLIP)
    bins = equal_count_bins(p)
    out = {}

    def stats(Wm):
        n = Wm.sum(1)
        prev = (Wm * y).sum(1) / n
        brier = (Wm * (p - y) ** 2).sum(1) / n
        brier_ref = (Wm * (prev[:, None] - y[None, :]) ** 2).sum(1) / n
        a1 = wlogit_intercept(x, y, Wm)
        a2, b2 = wlogit_slope(x, y, Wm)
        pb, yb, share = binned(y, p, bins, Wm)
        ece = np.nansum(share * np.abs(yb - pb), axis=1)
        ll = -(Wm * (y * np.log(pc) + (1 - y) * np.log(1 - pc))).sum(1) / n
        return {"brier": brier, "brier_prevalence": brier_ref, "brier_skill": 1 - brier / brier_ref,
                "calibration_intercept": a1, "calibration_slope": b2, "joint_intercept": a2,
                "ECE": ece, "log_loss": ll, "prevalence": prev}

    pt = stats(ones)
    bt = stats(W)
    for k in pt:
        out[k] = est(pt[k][0], bt[k])
    auprc_b = np.array([average_precision_score(y, p, sample_weight=W[i]) for i in range(W.shape[0])])
    auroc_b = np.array([roc_auc_score(y, p, sample_weight=W[i]) for i in range(W.shape[0])])
    out["AUPRC"] = est(average_precision_score(y, p), auprc_b)
    out["AUROC"] = est(roc_auc_score(y, p), auroc_b)
    pb, yb, share = binned(y, p, bins, ones)
    cnt = np.array([(bins == b).sum() for b in range(N_BINS)])
    out["reliability"] = {"mean_pred": pb[0], "obs_freq": yb[0], "count": cnt}
    out["_boot"] = {k: bt[k] for k in ("brier", "brier_skill", "calibration_intercept", "calibration_slope", "ECE", "log_loss")}
    out["_boot"]["AUPRC"] = auprc_b
    out["_boot"]["AUROC"] = auroc_b
    return out


def point_stats(y, p, thr):
    TP, FP, TN, FN = confusion(y, p, thr)
    ch = characteristics(TP, FP, TN, FN)
    return {"MCC": float(mcc_from(TP, FP, TN, FN)[0, 0]), "VME": float(ch["VME"][0, 0]), "ME": float(ch["ME"][0, 0])}


def cross_fitted(y, p, rep, fold):
    per_rep, per_fold = [], {"cross": {"MCC": [], "VME": [], "ME": []}, "half": {"MCC": [], "VME": [], "ME": []}}
    for r in range(3):
        o = rep != r
        TP, FP, TN, FN = confusion(y[o], p[o], GRID)
        mcc = mcc_from(TP, FP, TN, FN)[0]
        best = GRID[mcc >= mcc.max() - 1e-12]
        t_r = float(best[len(best) // 2])
        m = rep == r
        per_rep.append({"repeat": r, "threshold": t_r, "tuning_MCC": float(mcc.max()),
                        "cross": point_stats(y[m], p[m], t_r), "half": point_stats(y[m], p[m], 0.5)})
        for f in range(5):
            mf = m & (fold == f)
            for key, thr in (("cross", t_r), ("half", 0.5)):
                s = point_stats(y[mf], p[mf], thr)
                for k in ("MCC", "VME", "ME"):
                    per_fold[key][k].append(s[k])
    out = {"per_repeat": per_rep, "thresholds": [x["threshold"] for x in per_rep]}
    for key in ("cross", "half"):
        out[key] = {k: float(np.mean([x[key][k] for x in per_rep])) for k in ("MCC", "VME", "ME")}
        out[f"{key}_per_fold_mean"] = {k: float(np.mean(per_fold[key][k])) for k in ("MCC", "VME", "ME")}
    out["_per_fold"] = {key: {k: np.array(v) for k, v in per_fold[key].items()} for key in per_fold}
    return out


def vme_me_curve(y, p, thr):
    res, sus = y == 1, y == 0
    return np.array([np.mean(p[res] < t) for t in thr]), np.array([np.mean(p[sus] >= t) for t in thr])


def profile_errors(ytrue, prob, rep, fold, labels):
    methods = sorted(prob)
    wrong = {m: np.sum([(prob[m][d] >= 0.5).astype(float) != ytrue[d] for d in labels], axis=0) for m in methods}
    keys = [(r, f) for r in range(3) for f in range(5)]
    masks = [(rep == r) & (fold == f) for r, f in keys]
    per_method = {}
    for m in methods:
        w = wrong[m]
        per_method[m] = {
            "subset_accuracy": np.array([np.mean(w[mk] == 0) for mk in masks]),
            "hamming_loss": np.array([np.mean(w[mk]) / len(labels) for mk in masks]),
            "all_wrong": np.array([np.mean(w[mk] == 3) for mk in masks]),
            "at_least_two_wrong": np.array([np.mean(w[mk] >= 2) for mk in masks]),
        }
    out = {"per_method": {}, "versus_BR": {}, "pooled_error_structure": {}}
    wb = wrong["binary_relevance"]
    for m in methods:
        out["per_method"][m] = {k: {"mean": float(v.mean()), "per_fold": v} for k, v in per_method[m].items()}
        errs = np.array([(prob[m][d] >= 0.5).astype(float) != ytrue[d] for d in labels]).astype(float)
        rate = errs.mean(1)
        phi = {}
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                phi[f"{labels[i]}-{labels[j]}"] = float(np.corrcoef(errs[i], errs[j])[0, 1])
        p_all_ind = float(np.prod(rate))
        p_ge2_ind = float(sum(rate[i] * rate[j] * (1 - rate[k]) for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0))) + p_all_ind)
        aw = np.flatnonzero(wrong[m] == 3)
        comp = {}
        for i in aw:
            key = "".join(str(int(ytrue[d][i])) for d in labels)
            comp[key] = comp.get(key, 0) + 1
        out["pooled_error_structure"][m] = {
            "per_drug_error_rate": dict(zip(labels, rate.tolist())), "error_phi": phi,
            "all_wrong_observed": float(np.mean(wrong[m] == 3)), "all_wrong_if_independent": p_all_ind,
            "at_least_two_wrong_observed": float(np.mean(wrong[m] >= 2)), "at_least_two_wrong_if_independent": p_ge2_ind,
            "all_wrong_rows": int(aw.size),
            "all_wrong_true_profiles": dict(sorted(comp.items())),   # true CIP CTX CAZ profile -> pooled rows (prediction is its complement)
        }
        if m == "binary_relevance":
            continue
        w = wrong[m]
        br_ok_m_bad = np.array([np.mean((wb[mk] == 0) & (w[mk] >= 1)) for mk in masks])
        m_ok_br_bad = np.array([np.mean((w[mk] == 0) & (wb[mk] >= 1)) for mk in masks])
        blk = {}
        for k in per_method[m]:
            d = per_method[m][k] - per_method["binary_relevance"][k]
            obs, pv = exact_sign_flip_pvalue(d)
            blk[k] = {"delta_mean": obs, "p": pv, "per_fold_delta": d}
        obs, pv = exact_sign_flip_pvalue(br_ok_m_bad - m_ok_br_bad)
        blk["BR_correct_method_wrong"] = {"mean": float(br_ok_m_bad.mean()), "per_fold": br_ok_m_bad}
        blk["method_correct_BR_wrong"] = {"mean": float(m_ok_br_bad.mean()), "per_fold": m_ok_br_bad}
        blk["discordance_asymmetry"] = {"delta_mean": obs, "p": pv}
        out["versus_BR"][m] = blk
    return out


# figures are drawn from the stored blocks only, so they can be redrawn with --plot-only
def _ps():
    from mecor.visualization import paper_style as ps
    return ps


def _method_handles(blocks, ps):
    """Legend handles for BR, the chain ensemble and the best chain (named when it is the same ordering for every drug)."""
    from matplotlib.lines import Line2D
    chains = {blocks[d]["best_chain"] for d in DRUGS if not blocks[d]["best_equals_BR"]}
    best = "Best chain" + (f" ({chain_label(next(iter(chains))).replace('>', ' → ')})" if len(chains) == 1 else "")
    return [Line2D([], [], color=ps.ROLE[r], lw=1.2, label=l) for r, l in (("br", "Binary relevance"), ("ensemble", "Chain ensemble"), ("best", best))]


def _drug_text(ax, d, x=0.98, ha="right"):
    ax.text(x, 0.96, d, transform=ax.transAxes, ha=ha, va="top", fontsize=8, color=_ps().INK2)


def _axis_ticks(ax, xs, color, frac=0.045):
    """Short ticks rising from the bottom spine at the given x positions (break-even, prevalence, chosen thresholds)."""
    y0, y1 = ax.get_ylim()
    ax.vlines(xs, y0, y0 + frac * (y1 - y0), color=color, lw=1.2, zorder=4)


def stacked_dca():
    """Decision curves of the stacked model (full grid, scripts/04_stacking/01_nested_stacking.py) per drug on the pooled out-of-fold predictions of the
    same 15 folds, with the same isolate-level bootstrap resamples as the other curves (same seed and row order)."""
    cfg = Config.from_yaml(str(ROOT / "configs/s_orsola.yaml")); X, Y = load_dataset(cfg)
    folds = reconstruct_folds(X, Y); keys = sorted(folds); iso = np.concatenate([folds[k] for k in keys])
    Z = [np.load(STACK_DIR / f"r{r}_f{f}.npz") for r, f in keys]
    W = bootstrap_weights(len(Y), iso, np.random.default_rng(SEED))
    out = {}
    for j, d in enumerate(DRUGS):
        y = np.concatenate([z["y"][:, j] for z in Z]).astype(float)
        assert np.array_equal(y, Y.iloc[iso][d].values.astype(float)), "stacked fold alignment mismatch"
        nb, nbb = decision_curve(y, np.concatenate([z["stack"][:, j] for z in Z]), W)
        out[d] = {"nb": nb, "nb_ci": [np.percentile(nbb, 2.5, axis=0), np.percentile(nbb, 97.5, axis=0)]}
    return out


def fig_dca(model, dca, stack=None):
    """Net benefit against threshold probability, one panel per drug; treat-all and treat-none as neutral references.
    With stack (logistic regression): the stacked model replaces the best single chain, bands for BR and the stacked model."""
    from matplotlib.lines import Line2D
    ps = _ps()
    fig, axes = ps.figure("double", height=2.7, nrows=1, ncols=3, sharey=True, constrained_layout=True)
    for ax, d, letter in zip(axes, DRUGS, "abc"):
        blk = dict(dca[d]); thr = np.asarray(blk["thresholds"], float); prev = float(blk["prevalence"])
        ps.hline(ax, 0, color=ps.MUTED, lw=0.8, ls=":")
        ax.plot(thr, np.asarray(blk["treat_all"]["nb"], float), color=ps.MUTED, lw=1.0, ls="--", zorder=1)
        if stack is not None: blk["Stack"] = stack[d]
        series = (("BR", "br", True), ("Ens", "ensemble", False), ("Stack", "stacking", True)) if stack is not None else \
                 (("BR", "br", True), ("Ens", "ensemble", True), ("Best", "best", False))
        for key, role, band in series:
            if key == "Best" and blk["best_equals_BR"]:
                continue
            c = ps.ROLE[role]
            ax.plot(thr, np.asarray(blk[key]["nb"], float), color=c, lw=1.2, zorder=3 if key in ("Ens", "Stack") else 2)
            if band:
                lo, hi = (np.asarray(v, float) for v in blk[key]["nb_ci"])
                ax.fill_between(thr, lo, hi, color=c, alpha=0.10, lw=0, zorder=1)
        ax.set_xlim(thr[0], thr[-1]); ax.set_ylim(-0.06, 0.45)
        _axis_ticks(ax, [prev], ps.INK2)
        ps.despine(ax, grid=None); ax.set_xlabel("Threshold probability"); _drug_text(ax, d); ps.panel_label(ax, letter)
    axes[0].set_ylabel("Net benefit per isolate")
    methods = _method_handles(dca, ps) if stack is None else \
        [Line2D([], [], color=ps.ROLE[r], lw=1.2, label=l) for r, l in (("br", "Binary relevance"), ("ensemble", "Chain ensemble"), ("stacking", "Stacked model"))]
    handles = [Line2D([], [], color=ps.MUTED, lw=0.8, ls=":", label="Treat none"), Line2D([], [], color=ps.MUTED, lw=1.0, ls="--", label="Treat all")] + methods + \
              [Line2D([], [], color=ps.INK2, ls="none", marker="|", ms=7, mew=1.2, label="Cohort prevalence")]
    ps.legend_above(fig, handles, [h.get_label() for h in handles], y=1.0)
    ps.save(fig, FIG_DIR / f"clinical_decision_dca_{model}")


def fig_calibration(model, cal):
    """Reliability curves (10 equal-count bins) with the identity line, one panel per drug; BSS and slope as in-panel text."""
    from matplotlib.lines import Line2D
    ps = _ps()
    fig, axes = ps.figure("double", height=2.8, nrows=1, ncols=3, sharey=True, constrained_layout=True)
    for ax, d, letter in zip(axes, DRUGS, "abc"):
        ax.plot([0, 1], [0, 1], color=ps.LIGHT, lw=0.8, ls="--", zorder=1)
        lines = []
        for key, role, name in (("BR", "br", "BR"), ("Ens", "ensemble", "Ensemble"), ("Best", "best", "Best chain")):
            if key == "Best" and cal[d]["best_equals_BR"]:
                continue
            blk = cal[d][key]; rel = blk["reliability"]
            ax.plot(rel["mean_pred"], rel["obs_freq"], marker="o", ms=4, lw=1.2, color=ps.ROLE[role], mec=ps.ROLE[role], zorder=3 if key == "Ens" else 2)
            lines.append(f"{name}: BSS {blk['brier_skill']['est']:+.2f}, slope {blk['calibration_slope']['est']:.2f}")
        ax.text(0.98, 0.03, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=ps.INK2, linespacing=1.4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_anchor("N"); ps.despine(ax, grid=None); ax.set_xlabel("Mean predicted probability"); _drug_text(ax, d, x=0.03, ha="left"); ps.panel_label(ax, letter, dy=1.04)
    axes[0].set_ylabel("Observed resistance frequency")
    handles = _method_handles(cal, ps) + [Line2D([], [], color=ps.LIGHT, lw=0.8, ls="--", label="Perfect calibration")]
    ps.legend_above(fig, handles, [h.get_label() for h in handles], y=1.0)
    ps.save(fig, FIG_DIR / f"clinical_decision_calibration_{model}")


def fig_vme_me(model, thr):
    """VME and ME against the decision threshold, one panel per drug; the 0.5 point is marked on every curve and the
    cross-fitted MCC-optimal thresholds are short ticks on the x axis."""
    from matplotlib.lines import Line2D
    ps = _ps()
    fig, axes = ps.figure("double", height=2.7, nrows=1, ncols=3, sharey=True, constrained_layout=True)
    for ax, d, letter in zip(axes, DRUGS, "abc"):
        sweep = thr[d]["vme_me_sweep"]; t = np.asarray(sweep["thresholds"], float); i5 = int(np.argmin(np.abs(t - 0.5)))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.08)
        ax.axvline(0.5, ymax=1.0 / 1.08, color=ps.LIGHT, lw=0.8, zorder=1)
        for key, role in (("BR", "br"), ("Ens", "ensemble"), ("Best", "best")):
            if key == "Best" and thr[d]["best_equals_BR"]:
                continue
            c = ps.ROLE[role]; z = 3 if key == "Ens" else 2
            for err, ls, hollow in (("VME", "-", False), ("ME", "--", True)):
                y = np.asarray(sweep[key][err], float)
                ax.plot(t, y, color=c, lw=1.2, ls=ls, zorder=z)
                ax.plot([t[i5]], [y[i5]], marker="o", ms=4.5, color=c, mfc=ps.SURFACE if hollow else c, mec=c, mew=1.0, ls="none", zorder=z + 2)
            _axis_ticks(ax, thr[d][key]["thresholds"], c)
        ps.despine(ax, grid=None); ax.set_xlabel("Decision threshold"); ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.text(0.5, 0.97, d, transform=ax.transAxes, ha="center", va="top", fontsize=8, color=ps.INK2); ps.panel_label(ax, letter)
    axes[0].set_ylabel("Error rate")
    handles = _method_handles(thr, ps) + [Line2D([], [], color=ps.INK, lw=1.2, ls="-", marker="o", ms=4.5, label="VME (resistant called susceptible)"),
                                          Line2D([], [], color=ps.INK, lw=1.2, ls="--", marker="o", ms=4.5, mfc=ps.SURFACE, label="ME (susceptible called resistant)"),
                                          Line2D([], [], color=ps.INK, ls="none", marker="|", ms=7, mew=1.2, label="Cross-fitted MCC-optimal threshold")]
    ps.legend_above(fig, handles, [h.get_label() for h in handles], ncol=3, y=1.0)
    ps.save(fig, FIG_DIR / f"clinical_decision_vme_me_{model}")


def plot_from_results(models=None, path=ROOT / "results/stats/clinical_decision.json"):
    """Redraw the three figures per model from the stored blocks (no bootstrap, no refitting)."""
    res = json.loads(Path(path).read_text())["models"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for model in (models or res):
        if model not in res:
            print(f"[skip] {model}: not in {path}")
            continue
        stack = None
        if model == "logistic" and STACK_DIR.exists():   # stacked decision curves, stored next to the other blocks
            stack = stacked_dca()
            (ROOT / "results/stats/clinical_decision_stacking_dca.json").write_text(json.dumps(jsonable({"thresholds": DCA_THR, "n_boot": N_BOOT, "stack": stack}), indent=1))
        fig_dca(model, res[model]["dca"], stack); fig_calibration(model, res[model]["calibration"]); fig_vme_me(model, res[model]["threshold"])
        print(f"figures redrawn for {model} in {FIG_DIR}")


def analyse_model(model, folds, Y, best_chains):
    cv = load_cv_results(ROOT / f"results/s_orsola/{model}/cv_results.pkl")
    labels = list(cv.labels)
    iso, rep, fold, ytrue, prob = pool(cv, folds, Y)
    rng = np.random.default_rng(SEED)
    W = bootstrap_weights(len(Y), iso, rng)
    out = {"n_rows": int(iso.size), "n_isolates": int(len(Y)), "best_chain": {}, "dca": {}, "operating": {},
           "calibration": {}, "threshold": {}, "profile": None}
    prev_by_drug = {d: float(ytrue[d].mean()) for d in DRUGS}
    for d in DRUGS:
        bc = best_chains[model][d]
        best_eq_br = bc.split("_")[1] == d
        out["best_chain"][d] = {"name": bc, "equals_BR": best_eq_br}
        meths = {"BR": "binary_relevance", "Ens": "ensemble", "Best": bc}
        y = ytrue[d]
        dca = {"prevalence": prev_by_drug[d], "best_chain": bc, "best_equals_BR": best_eq_br, "thresholds": DCA_THR}
        nb_all, nbb_all = treat_all_curve(y, W)
        dca["treat_all"] = {"nb": nb_all, "nb_ci": [np.percentile(nbb_all, 2.5, axis=0), np.percentile(nbb_all, 97.5, axis=0)]}
        dca["treat_none"] = {"nb": np.zeros_like(DCA_THR)}
        nbb = {}
        for key, m in meths.items():
            nb, nbb[key] = decision_curve(y, prob[m][d], W)
            dca[key] = {"nb": nb, "nb_ci": [np.percentile(nbb[key], 2.5, axis=0), np.percentile(nbb[key], 97.5, axis=0)]}
        dca["Ens_minus_BR"] = {"delta": dca["Ens"]["nb"] - dca["BR"]["nb"],
                               "ci": [np.percentile(nbb["Ens"] - nbb["BR"], 2.5, axis=0), np.percentile(nbb["Ens"] - nbb["BR"], 97.5, axis=0)],
                               "p_boot_Ens_gt_BR": (nbb["Ens"] > nbb["BR"]).mean(0)}
        dca["ranges"] = {
            "Ens_gt_BR": ranges(dca["Ens"]["nb"] > dca["BR"]["nb"] + 1e-12),
            "BR_gt_treat_all": ranges(dca["BR"]["nb"] > nb_all + 1e-12),
            "Ens_gt_treat_all": ranges(dca["Ens"]["nb"] > nb_all + 1e-12),
            "Best_gt_treat_all": ranges(dca["Best"]["nb"] > nb_all + 1e-12),
            "Ens_gt_treat_all_boot95": ranges(np.percentile(nbb["Ens"] - nbb_all, 2.5, axis=0) > 0),
            "BR_gt_treat_all_boot95": ranges(np.percentile(nbb["BR"] - nbb_all, 2.5, axis=0) > 0),
        }
        rep_rows = {}
        for t in REPORT_THR + [round(prev_by_drug[d], 2)]:
            i = int(np.argmin(np.abs(DCA_THR - t)))
            rep_rows[f"{DCA_THR[i]:.2f}"] = {
                "treat_all": est(nb_all[i], nbb_all[:, i]), "treat_none": {"est": 0.0, "ci": [0.0, 0.0]},
                **{key: est(dca[key]["nb"][i], nbb[key][:, i]) for key in meths},
                "Ens_minus_BR": est(dca["Ens"]["nb"][i] - dca["BR"]["nb"][i], nbb["Ens"][:, i] - nbb["BR"][:, i]),
                "p_boot_Ens_gt_BR": float((nbb["Ens"][:, i] > nbb["BR"][:, i]).mean()),
                "p_boot_Ens_gt_treat_all": float((nbb["Ens"][:, i] > nbb_all[:, i]).mean()),
                "p_boot_BR_gt_treat_all": float((nbb["BR"][:, i] > nbb_all[:, i]).mean()),
            }
        dca["at_thresholds"] = rep_rows
        out["dca"][d] = dca
        opr = {"best_chain": bc, "best_equals_BR": best_eq_br}
        boots = {}
        for key, m in meths.items():
            p = prob[m][d]
            blk = {"half": operating_block(y, p, W, 0.5)}
            for tgt in VME_TARGETS:
                blk[f"vme_{tgt:.2f}"] = operating_block(y, p, W, vme_threshold(y, p, tgt))
            rt = rule_thresholds(y, p)
            ss = scheme_stats(y, p, rt["thr_in"], rt["thr_out"], np.ones((1, y.size)))
            ssb = scheme_stats(y, p, rt["thr_in"], rt["thr_out"], W)
            blk["rule_in_out"] = {**rt, **{k: est(ss[k][0], ssb[k]) for k in ss}}
            boots[key] = {k: blk[k].pop("_boot") for k in ("half", "vme_0.10", "vme_0.15")}
            boots[key]["rule"] = ssb
            opr[key] = blk
        # chain minus BR (paired bootstrap)
        delta = {}
        for key in ("Ens", "Best"):
            dk = {}
            for op in ("half", "vme_0.10", "vme_0.15"):
                dd = {}
                for k in ("sensitivity", "specificity", "PPV", "NPV", "LR_pos", "LR_neg"):
                    dd[k] = est(opr[key][op][k]["est"] - opr["BR"][op][k]["est"], boots[key][op][k] - boots["BR"][op][k])
                for prev in PREVALENCES:
                    pk = f"{prev:.2f}"
                    for call, lr in (("after_resistant_call", "LR_pos"), ("after_susceptible_call", "LR_neg")):
                        dd[f"post_test_{pk}_{call}"] = est(
                            opr[key][op]["post_test"][pk][call]["est"] - opr["BR"][op]["post_test"][pk][call]["est"],
                            post_test(prev, boots[key][op][lr]) - post_test(prev, boots["BR"][op][lr]))
                dk[op] = dd
            dk["rule_in_out"] = {k: est(opr[key]["rule_in_out"][k]["est"] - opr["BR"]["rule_in_out"][k]["est"],
                                        boots[key]["rule"][k] - boots["BR"]["rule"][k])
                                 for k in ("frac_actionable", "PPV_rule_in", "NPV_rule_out", "coverage_resistant")}
            delta[f"{key}_minus_BR"] = dk
        opr["delta"] = delta
        out["operating"][d] = opr
        cal = {"best_chain": bc, "best_equals_BR": best_eq_br}
        cboot = {}
        for key, m in meths.items():
            cal[key] = calibration_block(y, prob[m][d], W)
            cboot[key] = cal[key].pop("_boot")
        cal["delta"] = {}
        for key in ("Ens", "Best"):
            cal["delta"][f"{key}_minus_BR"] = {
                k: est(cal[key][k]["est"] - cal["BR"][k]["est"], cboot[key][k] - cboot["BR"][k]) for k in cboot[key]}
            cal["delta"][f"{key}_minus_BR"]["p_boot_brier_lower"] = float((cboot[key]["brier"] < cboot["BR"]["brier"]).mean())
        out["calibration"][d] = cal
        thr = {"best_chain": bc, "best_equals_BR": best_eq_br}
        pf = {}
        for key, m in meths.items():
            thr[key] = cross_fitted(y, prob[m][d], rep, fold)
            pf[key] = thr[key].pop("_per_fold")
        thr["delta"] = {}
        for key in ("Ens", "Best"):
            dk = {}
            for which in ("cross", "half"):
                for k in ("MCC", "VME", "ME"):
                    obs, pv = exact_sign_flip_pvalue(pf[key][which][k] - pf["BR"][which][k])
                    dk[f"{which}_{k}"] = {"delta_per_fold_mean": obs, "p": pv}
            thr["delta"][f"{key}_minus_BR"] = dk
        sweep = np.round(np.arange(0, 1.0 + 1e-9, 0.01), 2)
        thr["vme_me_sweep"] = {"thresholds": sweep}
        for key, m in meths.items():
            v, me = vme_me_curve(y, prob[m][d], sweep)
            thr["vme_me_sweep"][key] = {"VME": v, "ME": me}
        out["threshold"][d] = thr
    out["profile"] = profile_errors(ytrue, prob, rep, fold, labels)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_dca(model, out["dca"])
    fig_calibration(model, out["calibration"])
    fig_vme_me(model, out["threshold"])
    return out


def chain_label(name):
    return name.replace("chain_", "").replace("_", ">")


def tables(res):
    L = ["# Clinical-decision bundle, Sant'Orsola cohort, pooled out-of-fold predictions", "",
         "All ratios are computed on the pooled out-of-fold predictions of the 5x3 repeated nested CV (561 rows, "
         "187 isolates x 3 repeats). Intervals are 95 percent isolate-bootstrap percentile intervals (1000 resamples; "
         "the three predictions of an isolate are resampled together; the same resamples are used for all methods, so "
         "chain-minus-BR differences are paired). Best = MCC-best ordering per drug; for CIP it is identical to BR "
         "(CIP is the first link).", ""]
    for model in MODELS:
        if model not in res:
            continue
        R = res[model]
        name = MODEL_NAMES[model]
        L += [f"## {name}", ""]
        L += [f"###. Net benefit at selected threshold probabilities ({name})", "",
              "| Drug | Threshold | treat all | BR | Ens | Best | Ens minus BR [CI] | P(Ens > BR) | P(Ens > treat all) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for d in DRUGS:
            dca = R["dca"][d]
            for t in sorted(dca["at_thresholds"], key=float):
                row = dca["at_thresholds"][t]
                bl = "= BR" if dca["best_equals_BR"] else fci(row["Best"])
                L.append(f"| {d} | {t} | {fci(row['treat_all'])} | {fci(row['BR'])} | {fci(row['Ens'])} | {bl} | "
                         f"{fci(row['Ens_minus_BR'])} | {row['p_boot_Ens_gt_BR']:.3f} | {row['p_boot_Ens_gt_treat_all']:.3f} |")
        L += ["", "Threshold ranges (point estimates on the 0.05 to 0.60 grid):", "",
              "| Drug | Ens > BR | BR > treat all | Ens > treat all | Best > treat all | Ens > treat all (bootstrap 95% lower bound > 0) |",
              "|---|---|---|---|---|---|"]
        for d in DRUGS:
            rg = R["dca"][d]["ranges"]
            fmt = lambda v: ", ".join(f"{a:.2f} to {b:.2f}" for a, b in v) if v else "none"
            L.append(f"| {d} | {fmt(rg['Ens_gt_BR'])} | {fmt(rg['BR_gt_treat_all'])} | {fmt(rg['Ens_gt_treat_all'])} | "
                     f"{fmt(rg['Best_gt_treat_all'])} | {fmt(rg['Ens_gt_treat_all_boot95'])} |")
        L += ["", f"###. Operating characteristics ({name})", "",
              "| Drug | Method | Operating point | thr | Sens | Spec | PPV | NPV | LR+ | LR- |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for d in DRUGS:
            opr = R["operating"][d]
            for key in ("BR", "Ens", "Best"):
                if key == "Best" and opr["best_equals_BR"]:
                    continue
                lab = key if key != "Best" else f"Best ({chain_label(opr['best_chain'])})"
                for op, opl in (("half", "0.5"), ("vme_0.10", "VME 0.10"), ("vme_0.15", "VME 0.15")):
                    b = opr[key][op]
                    L.append(f"| {d} | {lab} | {opl} | {b['threshold']:.3f} | {fci(b['sensitivity'])} | {fci(b['specificity'])} | "
                             f"{fci(b['PPV'])} | {fci(b['NPV'])} | {fci(b['LR_pos'])} | {fci(b['LR_neg'])} |")
        L += ["", f"Two-threshold rule-in / rule-out scheme ({name}; rule-in when PPV first reaches 0.80, rule-out when NPV "
              "first reaches 0.90, otherwise the highest achievable value with at least 5 percent support):", "",
              "| Drug | Method | thr in | PPV target met | thr out | NPV target met | PPV (rule-in) | NPV (rule-out) | "
              "rule-in frac | rule-out frac | actionable frac | resistant covered |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for d in DRUGS:
            opr = R["operating"][d]
            for key in ("BR", "Ens", "Best"):
                if key == "Best" and opr["best_equals_BR"]:
                    continue
                r = opr[key]["rule_in_out"]
                lab = key if key != "Best" else f"Best ({chain_label(opr['best_chain'])})"
                L.append(f"| {d} | {lab} | {r['thr_in']:.2f} | {'yes' if r['ppv_target_met'] else 'no'} | {r['thr_out']:.2f} | "
                         f"{'yes' if r['npv_target_met'] else 'no'} | {fci(r['PPV_rule_in'])} | {fci(r['NPV_rule_out'])} | "
                         f"{f3(r['frac_rule_in']['est'])} | {f3(r['frac_rule_out']['est'])} | {fci(r['frac_actionable'])} | "
                         f"{f3(r['coverage_resistant']['est'])} |")
        L += ["", f"Post-test probability of resistance after a resistant call (R) and after a susceptible call (S) ({name}):", "",
              "| Drug | Method | Operating point | " + " | ".join(f"prev {p:.2f}: R / S" for p in PREVALENCES) + " |",
              "|---|---|---|" + "---|" * len(PREVALENCES)]
        for d in DRUGS:
            opr = R["operating"][d]
            for key in ("BR", "Ens", "Best"):
                if key == "Best" and opr["best_equals_BR"]:
                    continue
                lab = key if key != "Best" else f"Best ({chain_label(opr['best_chain'])})"
                for op, opl in (("half", "0.5"), ("vme_0.15", "VME 0.15")):
                    pt = opr[key][op]["post_test"]
                    cells = " | ".join(f"{f3(pt[f'{p:.2f}']['after_resistant_call']['est'])} / {f3(pt[f'{p:.2f}']['after_susceptible_call']['est'])}"
                                       for p in PREVALENCES)
                    L.append(f"| {d} | {lab} | {opl} | {cells} |")
        L += ["", f"Chain ensemble minus BR, paired bootstrap ({name}):", "",
              "| Drug | Operating point | dSens | dSpec | dPPV | dNPV | dLR+ | dLR- | d post-test R at 0.42 | d post-test S at 0.42 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for d in DRUGS:
            dl = R["operating"][d]["delta"]["Ens_minus_BR"]
            for op, opl in (("half", "0.5"), ("vme_0.15", "VME 0.15")):
                b = dl[op]
                L.append(f"| {d} | {opl} | {fci(b['sensitivity'])} | {fci(b['specificity'])} | {fci(b['PPV'])} | {fci(b['NPV'])} | "
                         f"{fci(b['LR_pos'])} | {fci(b['LR_neg'])} | {fci(b['post_test_0.42_after_resistant_call'])} | "
                         f"{fci(b['post_test_0.42_after_susceptible_call'])} |")
        L += ["", f"###. Calibration ({name})", "",
              "| Drug | Method | Brier | Brier (prevalence) | Brier skill | Intercept | Slope | ECE | Log-loss | AUPRC | AUROC |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for d in DRUGS:
            cal = R["calibration"][d]
            for key in ("BR", "Ens", "Best"):
                if key == "Best" and cal["best_equals_BR"]:
                    continue
                c = cal[key]
                lab = key if key != "Best" else f"Best ({chain_label(cal['best_chain'])})"
                L.append(f"| {d} | {lab} | {fci(c['brier'])} | {f3(c['brier_prevalence']['est'])} | {fci(c['brier_skill'])} | "
                         f"{fci(c['calibration_intercept'])} | {fci(c['calibration_slope'])} | {fci(c['ECE'])} | "
                         f"{fci(c['log_loss'])} | {fci(c['AUPRC'])} | {fci(c['AUROC'])} |")
        L += ["", "Chain ensemble minus BR (paired bootstrap):", "",
              "| Drug | dBrier | dBrier skill | dIntercept | dSlope | dECE | dLog-loss | dAUPRC | P(Brier Ens < BR) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for d in DRUGS:
            b = R["calibration"][d]["delta"]["Ens_minus_BR"]
            L.append(f"| {d} | {fci(b['brier'])} | {fci(b['brier_skill'])} | {fci(b['calibration_intercept'])} | {fci(b['calibration_slope'])} | "
                     f"{fci(b['ECE'])} | {fci(b['log_loss'])} | {fci(b['AUPRC'])} | {b['p_boot_brier_lower']:.3f} |")
        L += ["", f"###. Fixed 0.5 versus cross-fitted MCC-optimal threshold ({name}; per-repeat pooled values, mean over the three repeats; "
              "sign-flip p over the 15 outer folds for Ens minus BR)", "",
              "| Drug | Method | thresholds (3 repeats) | MCC at 0.5 | MCC cross-fitted | VME at 0.5 | VME cross-fitted | ME at 0.5 | ME cross-fitted |",
              "|---|---|---|---|---|---|---|---|---|"]
        for d in DRUGS:
            th = R["threshold"][d]
            for key in ("BR", "Ens", "Best"):
                if key == "Best" and th["best_equals_BR"]:
                    continue
                t = th[key]
                lab = key if key != "Best" else f"Best ({chain_label(th['best_chain'])})"
                L.append(f"| {d} | {lab} | {', '.join(f'{x:.2f}' for x in t['thresholds'])} | {f3(t['half']['MCC'])} | {f3(t['cross']['MCC'])} | "
                         f"{f3(t['half']['VME'])} | {f3(t['cross']['VME'])} | {f3(t['half']['ME'])} | {f3(t['cross']['ME'])} |")
            for key in ("Ens", "Best"):
                if key == "Best" and th["best_equals_BR"]:
                    continue
                dl = th["delta"][f"{key}_minus_BR"]
                L.append(f"| {d} | {key} minus BR (per-fold mean, p) | | {dl['half_MCC']['delta_per_fold_mean']:+.3f} (p {fp(dl['half_MCC']['p'])}) | "
                         f"{dl['cross_MCC']['delta_per_fold_mean']:+.3f} (p {fp(dl['cross_MCC']['p'])}) | "
                         f"{dl['half_VME']['delta_per_fold_mean']:+.3f} (p {fp(dl['half_VME']['p'])}) | "
                         f"{dl['cross_VME']['delta_per_fold_mean']:+.3f} (p {fp(dl['cross_VME']['p'])}) | "
                         f"{dl['half_ME']['delta_per_fold_mean']:+.3f} (p {fp(dl['half_ME']['p'])}) | "
                         f"{dl['cross_ME']['delta_per_fold_mean']:+.3f} (p {fp(dl['cross_ME']['p'])}) |")
        pr = R["profile"]
        L += ["", f"###. Profile errors per outer fold, 0.5 threshold ({name}; means over 15 folds; sign-flip p for method minus BR)", "",
              "| Method | Subset accuracy | Hamming loss | All three wrong | At least two wrong | BR correct, method wrong | Method correct, BR wrong | asymmetry p |",
              "|---|---|---|---|---|---|---|---|"]
        pm = pr["per_method"]["binary_relevance"]
        L.append(f"| BR | {f3(pm['subset_accuracy']['mean'])} | {f3(pm['hamming_loss']['mean'])} | {f3(pm['all_wrong']['mean'])} | "
                 f"{f3(pm['at_least_two_wrong']['mean'])} | | | |")
        for m in ["ensemble"] + sorted(k for k in pr["versus_BR"] if k.startswith("chain")):
            pm, vs = pr["per_method"][m], pr["versus_BR"][m]
            lab = "Ens" if m == "ensemble" else chain_label(m)
            L.append(f"| {lab} | {f3(pm['subset_accuracy']['mean'])} ({vs['subset_accuracy']['delta_mean']:+.3f}, p {fp(vs['subset_accuracy']['p'])}) | "
                     f"{f3(pm['hamming_loss']['mean'])} ({vs['hamming_loss']['delta_mean']:+.3f}, p {fp(vs['hamming_loss']['p'])}) | "
                     f"{f3(pm['all_wrong']['mean'])} ({vs['all_wrong']['delta_mean']:+.3f}, p {fp(vs['all_wrong']['p'])}) | "
                     f"{f3(pm['at_least_two_wrong']['mean'])} ({vs['at_least_two_wrong']['delta_mean']:+.3f}, p {fp(vs['at_least_two_wrong']['p'])}) | "
                     f"{f3(vs['BR_correct_method_wrong']['mean'])} | {f3(vs['method_correct_BR_wrong']['mean'])} | {fp(vs['discordance_asymmetry']['p'])} |")
        L += ["", "Pooled error structure (phi between per-drug error indicators; observed versus independence-expected fraction of profiles with all three or at least two calls wrong):", "",
              "| Method | err CIP | err CTX | err CAZ | phi CIP-CTX | phi CIP-CAZ | phi CTX-CAZ | all wrong obs / indep | >= 2 wrong obs / indep | true profiles (CIP CTX CAZ) of the all-wrong rows |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for m in ["binary_relevance", "ensemble"]:
            e = pr["pooled_error_structure"][m]
            lab = "BR" if m == "binary_relevance" else "Ens"
            comp = ", ".join(f"{k}: {v}" for k, v in e["all_wrong_true_profiles"].items())
            L.append(f"| {lab} | {f3(e['per_drug_error_rate']['CIP'])} | {f3(e['per_drug_error_rate']['CTX'])} | {f3(e['per_drug_error_rate']['CAZ'])} | "
                     f"{f3(e['error_phi']['CIP-CTX'])} | {f3(e['error_phi']['CIP-CAZ'])} | {f3(e['error_phi']['CTX-CAZ'])} | "
                     f"{f3(e['all_wrong_observed'])} / {f3(e['all_wrong_if_independent'])} | "
                     f"{f3(e['at_least_two_wrong_observed'])} / {f3(e['at_least_two_wrong_if_independent'])} | {comp} (n = {e['all_wrong_rows']}) |")
        L.append("")
    L += ["## Cross-model summary (BR / Ens)", "",
          "| Model | Drug | NB at 0.42: treat all / BR / Ens | Ens minus BR at 0.42 [CI] | NB at 0.50: treat all / BR / Ens | "
          "Brier skill BR / Ens | dBrier Ens minus BR [CI] | MCC cross-fitted BR / Ens (p) |", "|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        if model not in res:
            continue
        R = res[model]
        for d in DRUGS:
            a42, a50 = R["dca"][d]["at_thresholds"]["0.42"], R["dca"][d]["at_thresholds"]["0.50"]
            cal, th = R["calibration"][d], R["threshold"][d]
            L.append(f"| {MODEL_NAMES[model]} | {d} | {f3(a42['treat_all']['est'])} / {f3(a42['BR']['est'])} / {f3(a42['Ens']['est'])} | "
                     f"{fci(a42['Ens_minus_BR'])} | {f3(a50['treat_all']['est'])} / {f3(a50['BR']['est'])} / {f3(a50['Ens']['est'])} | "
                     f"{f3(cal['BR']['brier_skill']['est'])} / {f3(cal['Ens']['brier_skill']['est'])} | {fci(cal['delta']['Ens_minus_BR']['brier'])} | "
                     f"{f3(th['BR']['cross']['MCC'])} / {f3(th['Ens']['cross']['MCC'])} (p {fp(th['delta']['Ens_minus_BR']['cross_MCC']['p'])}) |")
    L += ["", "Profile errors per model (0.5 threshold; BR / Ens, sign-flip p over 15 folds):", "",
          "| Model | Subset accuracy | Hamming loss | All three wrong | At least two wrong | BR correct, Ens wrong / Ens correct, BR wrong (p) | error phi CTX-CAZ |",
          "|---|---|---|---|---|---|---|"]
    for model in MODELS:
        if model not in res:
            continue
        pr = res[model]["profile"]
        pb, pe, vs = pr["per_method"]["binary_relevance"], pr["per_method"]["ensemble"], pr["versus_BR"]["ensemble"]
        es = pr["pooled_error_structure"]
        cells = " | ".join(f"{f3(pb[k]['mean'])} / {f3(pe[k]['mean'])} (p {fp(vs[k]['p'])})"
                           for k in ("subset_accuracy", "hamming_loss", "all_wrong", "at_least_two_wrong"))
        L.append(f"| {MODEL_NAMES[model]} | {cells} | {f3(vs['BR_correct_method_wrong']['mean'])} / {f3(vs['method_correct_BR_wrong']['mean'])} "
                 f"(p {fp(vs['discordance_asymmetry']['p'])}) | {f3(es['binary_relevance']['error_phi']['CTX-CAZ'])} / {f3(es['ensemble']['error_phi']['CTX-CAZ'])} |")
    return "\n".join(L) + "\n"


def main():
    cfg = Config.from_yaml(str(ROOT / "configs/s_orsola.yaml"))
    X, Y = load_dataset(cfg)
    folds = reconstruct_folds(X, Y)
    stats = json.loads(STATS_JSON.read_text())
    best_chains = {m: {d: stats[m]["drugs"][d]["MCC"]["best_chain"] for d in DRUGS} for m in MODELS}
    res = {}
    for model in MODELS:
        path = ROOT / f"results/s_orsola/{model}/cv_results.pkl"
        if not path.exists():
            print(f"[skip] {path}")
            continue
        print(f"== {model}: best chains by MCC {best_chains[model]}")
        res[model] = analyse_model(model, folds, Y, best_chains)
        print("   done")
    settings = {"seed": SEED, "n_boot": N_BOOT, "dca_thresholds": DCA_THR, "report_thresholds": REPORT_THR,
                "prevalences_for_post_test": PREVALENCES, "vme_targets": VME_TARGETS, "ppv_target": PPV_TARGET,
                "npv_target": NPV_TARGET, "min_support_frac": MIN_SUPPORT, "logit_clip": CLIP, "ece_bins": N_BINS,
                "note": "Pooled out-of-fold predictions (187 isolates x 3 repeats = 561 rows). Isolate bootstrap: "
                        "isolates resampled with replacement, the three rows of an isolate share the resample weight; "
                        "the same 1000 resamples are used for all methods and models. Calibration intercept: logistic "
                        "model with logit(p) as offset (slope fixed at 1); slope: logistic model y ~ a + b logit(p) "
                        "(joint_intercept is a). ECE bins are fixed from the full pooled set. Cross-fitted threshold: "
                        "MCC-maximizing threshold on the 0.01 grid chosen on the pooled predictions of the other two "
                        "repeats (median of ties); the labels of the isolates to which it is applied are therefore "
                        "seen through other repeats' predictions, a limitation of a design without a held-out split."}
    (ROOT / "results/stats").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/stats/clinical_decision.json").write_text(json.dumps(jsonable({"settings": settings, "models": res}), indent=1))
    (ROOT / "tables").mkdir(exist_ok=True)
    md = tables(res)
    (ROOT / "tables/clinical_decision.md").write_text(md)
    print(md)
    print("saved results/stats/clinical_decision.json, tables/clinical_decision.md and figures in figures/supplementary/")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plot-only", action="store_true", help="redraw the figures from results/stats/clinical_decision.json without recomputing")
    ap.add_argument("--model", nargs="*", default=None, help="with --plot-only: restrict to these models (default: every model in the JSON)")
    args = ap.parse_args()
    if args.plot_only:
        plot_from_results(args.model)
    else:
        main()
