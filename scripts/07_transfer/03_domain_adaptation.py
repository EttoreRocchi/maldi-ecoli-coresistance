"""Cross-site transfer with domain adaptation (logistic regression, reduced grid).

Source: DRIAMS-A (CIP, CRO, CAZ). Targets: Sant'Orsola (CIP, CTX as the ceftriaxone-class label, CAZ) and DRIAMS-B.
For each target, R random draws select k local labeled isolates (k in K_LIST) and the remaining target isolates form
the test set. Methods compared on the same test isolates:
  source_only        : trained on DRIAMS-A, applied as is
  source_zscore      : per-site standardization of every m/z bin (source and target scaled with their own statistics;
                       unsupervised, uses only target spectra)
  source_coral       : per-site standardization plus CORAL alignment of the source covariance to the target covariance
                       (unsupervised)
  source_recal       : source_only with the intercept recalibrated to the local prevalence estimated from the k labels
                       (prior shift; affects MCC and calibration, not AUROC)
  local_only         : trained on the k local isolates only
  source_plus_local  : trained on DRIAMS-A plus the k local isolates, the local ones weighted so that their total
                       weight equals the source's (label-efficient fine-tuning for a linear model)
For each method: binary relevance per drug and the a priori chain CIP -> ceph -> CAZ (hard and probability passing).
Outputs: results/stats/domain_adaptation.json, tables/domain_adaptation.md, figures/supplementary/domain_adaptation.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.config import Config  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402
from mecor.evaluation.cv import _build_search_estimator  # noqa: E402

K_LIST = [25, 50, 100, 150]
R_DRAWS = 10
LABELS = ["CIP", "CRO", "CAZ"]


def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {"AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan, "MCC": float(matthews_corrcoef(y, yhat))}


def zscore(X, mu=None, sd=None):
    mu = X.mean(0) if mu is None else mu; sd = X.std(0) + 1e-8 if sd is None else sd
    return (X - mu) / sd, mu, sd


def coral_transform(Zs, Zt, lam=1.0):
    """Align source covariance to target covariance (Sun, Feng and Saenko 2016): Zs @ Cs^-1/2 @ Ct^1/2."""
    d = Zs.shape[1]
    Cs = np.cov(Zs, rowvar=False) + lam * np.eye(d); Ct = np.cov(Zt, rowvar=False) + lam * np.eye(d)
    def msqrt(C, inv=False):
        w, V = np.linalg.eigh(C); w = np.clip(w, 1e-6, None); s = 1 / np.sqrt(w) if inv else np.sqrt(w)
        return (V * s) @ V.T
    return Zs @ msqrt(Cs, inv=True) @ msqrt(Ct)


def fit_models(cfg, Xtr, Ytr, w=None):
    """BR per drug plus the links of the a priori chain CIP -> CRO -> CAZ."""
    fp = {} if w is None else {"clf__sample_weight": w}
    br = {}
    for d in LABELS:
        est = _build_search_estimator("logistic", cfg); est.fit(Xtr, Ytr[d], **fp); br[d] = est
    l2 = _build_search_estimator("logistic", cfg); l2.fit(np.column_stack([Xtr, Ytr["CIP"]]), Ytr["CRO"], **fp)
    l3 = _build_search_estimator("logistic", cfg); l3.fit(np.column_stack([Xtr, Ytr["CIP"], Ytr["CRO"]]), Ytr["CAZ"], **fp)
    return br, l2, l3


def evaluate(models, Xte, Yte, shift=None):
    br, l2, l3 = models; rec = {}
    p = {d: br[d].predict_proba(Xte)[:, 1] for d in LABELS}
    if shift is not None:  # prior-shift intercept recalibration
        for d in LABELS:
            lg = np.log(np.clip(p[d], 1e-6, 1 - 1e-6) / (1 - np.clip(p[d], 1e-6, 1 - 1e-6))) + shift[d]; p[d] = 1 / (1 + np.exp(-lg))
    for d in LABELS: rec[f"br_{d}"] = metrics(Yte[d], p[d])
    for mode in ("hard", "proba"):
        u = (p["CIP"] >= 0.5).astype(float) if mode == "hard" else p["CIP"]
        p_cro = l2.predict_proba(np.column_stack([Xte, u]))[:, 1]
        m = (p_cro >= 0.5).astype(float) if mode == "hard" else p_cro
        p_caz = l3.predict_proba(np.column_stack([Xte, u, m]))[:, 1]
        rec[f"chain_{mode}_CRO"] = metrics(Yte["CRO"], p_cro); rec[f"chain_{mode}_CAZ"] = metrics(Yte["CAZ"], p_caz)
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", required=True, choices=["s_orsola", "driams_b"]); ap.add_argument("--n-jobs", type=int, default=24); ap.add_argument("--grid", default="liblinear", choices=["reduced", "liblinear", "full"]); args = ap.parse_args()
    cfg = Config.from_yaml(str(ROOT / "configs/driams_a.yaml")); cfg.model = "logistic"; cfg.param_search_mode = args.grid; cfg.n_jobs = args.n_jobs
    tag = "" if args.grid == "liblinear" else f"_{args.grid}"
    Xs_df, Ys_df = load_dataset(cfg); Xs = np.asarray(Xs_df, float); Ys = {d: Ys_df[d].values for d in LABELS}
    tcfg = Config.from_yaml(str(ROOT / f"configs/{args.target}.yaml")); tcfg.model = "logistic"
    Xt_df, Yt_df = load_dataset(tcfg); Xt = np.asarray(Xt_df, float)
    Yt_df = Yt_df.rename(columns={"CTX": "CRO"}); Yt = {d: Yt_df[d].values for d in LABELS}
    out_dir = ROOT / "results/domain_adaptation" / f"{args.target}{tag}"; out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda *a: print(*a, flush=True)
    # unsupervised variants (fit once)
    f = out_dir / "source_variants.json"
    if f.exists(): sv = json.load(open(f))
    else:
        sv = {}
        src = fit_models(cfg, Xs, Ys); sv["source_only"] = evaluate(src, Xt, Yt)
        prev_s = {d: Ys[d].mean() for d in LABELS}; prev_t = {d: Yt[d].mean() for d in LABELS}
        shift = {d: np.log(prev_t[d] / (1 - prev_t[d])) - np.log(prev_s[d] / (1 - prev_s[d])) for d in LABELS}
        sv["source_recal_oracle_prevalence"] = evaluate(src, Xt, Yt, shift=shift)
        Zs, _, _ = zscore(Xs); Zt, _, _ = zscore(Xt)
        zm = fit_models(cfg, Zs, Ys); sv["source_zscore"] = evaluate(zm, Zt, Yt)
        log("CORAL alignment ...")
        Zs_al = coral_transform(Zs, Zt); cm = fit_models(cfg, Zs_al, Ys); sv["source_coral"] = evaluate(cm, Zt, Yt)
        json.dump(sv, open(f, "w"), indent=1)
    for k, v in sv.items(): log(f"{args.target} {k}: " + ", ".join(f"{d} BR {v['br_'+d]['AUROC']:.3f}" for d in LABELS) + f", chain hard CRO {v['chain_hard_CRO']['AUROC']:.3f}")
    # label-efficiency draws
    strat = np.array(["".join(map(str, r)) for r in np.column_stack([Yt[d] for d in LABELS]).astype(int)])
    results = {}
    for k in K_LIST:
        for r in range(R_DRAWS):
            fk = out_dir / f"k{k}_draw{r}.json"
            if fk.exists(): results[(k, r)] = json.load(open(fk)); continue
            loc, te = train_test_split(np.arange(len(Xt)), train_size=k, stratify=strat, random_state=200 + r)
            Xl, Yl = Xt[loc], {d: Yt[d][loc] for d in LABELS}; Xte, Yte = Xt[te], {d: Yt[d][te] for d in LABELS}
            rec = {"k": k, "draw": r, "n_test": int(len(te))}
            if all(len(np.unique(Yl[d])) > 1 for d in LABELS):
                rec["local_only"] = evaluate(fit_models(cfg, Xt[loc], Yl), Xt[te], Yte)
            Xp = np.vstack([Xs, Xl]); Yp = {d: np.concatenate([Ys[d], Yl[d]]) for d in LABELS}
            w = np.concatenate([np.ones(len(Xs)), np.full(len(Xl), len(Xs) / len(Xl))])
            rec["source_plus_local"] = evaluate(fit_models(cfg, Xp, Yp, w=w), Xte, Yte)
            prev_l = {d: max(min(Yl[d].mean(), 0.99), 0.01) for d in LABELS}; prev_s = {d: Ys[d].mean() for d in LABELS}
            shift = {d: np.log(prev_l[d] / (1 - prev_l[d])) - np.log(prev_s[d] / (1 - prev_s[d])) for d in LABELS}
            rec["shift"] = shift
            json.dump(rec, open(fk, "w"), indent=1); results[(k, r)] = rec
            log(f"{args.target} k={k} draw {r}: local-only CRO {rec.get('local_only', {}).get('br_CRO', {}).get('AUROC', float('nan')):.3f}, source+local CRO {rec['source_plus_local']['br_CRO']['AUROC']:.3f}")
    rows = []
    for k in K_LIST:
        recs = [results[(k, r)] for r in range(R_DRAWS) if (k, r) in results]
        for meth in ("local_only", "source_plus_local"):
            vals = {}
            for key in ("br_CIP", "br_CRO", "br_CAZ", "chain_hard_CRO", "chain_proba_CRO", "chain_hard_CAZ", "chain_proba_CAZ"):
                a = [x[meth][key]["AUROC"] for x in recs if meth in x]; vals[key] = (float(np.nanmean(a)), float(np.nanstd(a))) if a else (np.nan, np.nan)
            rows.append({"k": k, "method": meth, "n_draws": sum(meth in x for x in recs), **{kk: v[0] for kk, v in vals.items()}, **{kk + "_sd": v[1] for kk, v in vals.items()}})
    summary = {"target": args.target, "source_variants": sv, "label_efficiency": rows, "n_source": int(len(Xs)), "n_target": int(len(Xt))}
    json.dump(summary, open(ROOT / f"results/stats/domain_adaptation_{args.target}{tag}.json", "w"), indent=1)
    lines = [f"# Domain adaptation, source DRIAMS-A (n = {len(Xs)}), target {args.target} (n = {len(Xt)}); logistic regression, {args.grid} grid", "",
             "## Unsupervised and source-only variants (whole target as test)", "", "| Method | CIP BR | CRO/CTX BR | CAZ BR | chain hard CRO | chain proba CRO | chain hard CAZ | CIP MCC | CRO MCC | CAZ MCC |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sv.items():
        lines.append(f"| {k} | {v['br_CIP']['AUROC']:.3f} | {v['br_CRO']['AUROC']:.3f} | {v['br_CAZ']['AUROC']:.3f} | {v['chain_hard_CRO']['AUROC']:.3f} | {v['chain_proba_CRO']['AUROC']:.3f} | {v['chain_hard_CAZ']['AUROC']:.3f} | {v['br_CIP']['MCC']:.3f} | {v['br_CRO']['MCC']:.3f} | {v['br_CAZ']['MCC']:.3f} |")
    lines += ["", "## Label efficiency (mean AUROC over draws, sd in parentheses)", "", "| k | Method | draws | CIP BR | CRO/CTX BR | CAZ BR | chain hard CRO | chain proba CRO |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['k']} | {r['method']} | {r['n_draws']} | {r['br_CIP']:.3f} ({r['br_CIP_sd']:.3f}) | {r['br_CRO']:.3f} ({r['br_CRO_sd']:.3f}) | {r['br_CAZ']:.3f} ({r['br_CAZ_sd']:.3f}) | {r['chain_hard_CRO']:.3f} | {r['chain_proba_CRO']:.3f} |")
    (ROOT / f"tables/domain_adaptation_{args.target}{tag}.md").write_text("\n".join(lines) + "\n"); log("\n".join(lines))


if __name__ == "__main__":
    main()
