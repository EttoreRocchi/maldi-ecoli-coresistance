"""Stratified analysis: discrimination and errors on the cephalosporin labels within the ciprofloxacin strata.

For each run of scripts/04_stacking/01_nested_stacking.py (results/nested_stacking/<tag>/r*_f*.npz: y, br, stack), the out-of-fold predictions of the
15 outer folds are pooled with their isolate identities (folds reconstructed as in the pipeline: 5 x 3 repeated stratified
folds on the joint profile, seed 42). Within the CIP-susceptible and the CIP-resistant isolates it reports, for each
cephalosporin label, the AUROC of every binary-relevance classifier, of the stacked model and, at Sant'Orsola, of the chain
ensemble (results/<site>/<model>/cv_results.pkl), and the very major (VME) and major (ME) error rates at 0.5.
Intervals: isolate bootstrap (2,000 resamples; the three out-of-fold rows of an isolate share a weight).
Outputs: results/stats/stratified_analysis.json, tables/stratified_analysis.md (table 0: AUROC within the strata for the
         cefotaxime or ceftriaxone label; table 1: error rates at 0.5 at Sant'Orsola; table 2: all labels and quantities).
         --tables-only rewrites the tables from the stored JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.config import Config  # noqa: E402
from mecor.cvio import load_cv_results  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402
from mecor.evaluation.grouping import load_patient_groups, outer_splits  # noqa: E402

SEED, N_BOOT = 42, 2000
RUNS = [("s_orsola_logistic_full", "s_orsola", "logistic", "Sant'Orsola", "logistic regression (full grid)"),
        ("s_orsola_catboost", "s_orsola", "catboost", "Sant'Orsola", "CatBoost"),
        ("s_orsola_tabpfn", "s_orsola", "tabpfn", "Sant'Orsola", "TabPFN"),
        ("driams_b_logistic", "driams_b", None, "DRIAMS-B", "logistic regression (liblinear grid)"),
        ("driams_c_logistic", "driams_c", None, "DRIAMS-C", "logistic regression (liblinear grid)"),
        ("driams_d_logistic", "driams_d", None, "DRIAMS-D", "logistic regression (liblinear grid)"),
        ("driams_a_logistic", "driams_a", None, "DRIAMS-A", "logistic regression (liblinear grid)")]
# --grouped: DRIAMS-A from the patient-grouped stacking run; outputs get the suffix _grouped (the DRIAMS-A analyses of the paper)
GROUPED = "--grouped" in sys.argv
SFX = "_grouped" if GROUPED else ""
if GROUPED:
    RUNS = [("driams_a_grouped_logistic", "driams_a_grouped", *rest) if t == "driams_a_logistic" else (t, s, *rest) for t, s, *rest in RUNS]


def pooled(tag, site, model):
    """Pool the out-of-fold rows of the 15 folds: isolate index, labels, BR, stacked and (optionally) ensemble probabilities."""
    cfg = Config.from_yaml(str(ROOT / f"configs/{site}.yaml")); X, Y = load_dataset(cfg); labels = list(Y.columns)
    strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
    groups = load_patient_groups(cfg, X.index)   # None unless the site config groups by patient: then the same folds as scripts/04_stacking/01_nested_stacking.py
    folds = {(g // 5, g % 5): te for g, (tr, te) in enumerate(outer_splits(strat.values, groups, 5, 3, SEED))}
    ens = None
    if model is not None:
        cv = load_cv_results(ROOT / f"results/{site}/{model}/cv_results.pkl")
        ens = {(fr.repeat_idx, fr.fold_idx): fr for fr in cv.fold_results if fr.method == "ensemble"}
    iso, y, br, st, en = [], [], [], [], []
    for k in sorted(folds):
        z = np.load(ROOT / f"results/nested_stacking/{tag}/r{k[0]}_f{k[1]}.npz")
        assert np.array_equal(z["y"], Y.iloc[folds[k]].values.astype(int)), f"fold reconstruction mismatch {tag} {k}"
        iso.append(folds[k]); y.append(z["y"]); br.append(z["br"]); st.append(z["stack"])
        if ens is not None:
            fr = ens[k]
            assert np.array_equal(np.column_stack([np.asarray(fr.y_true[d]) for d in labels]).astype(int), z["y"]), f"ensemble mismatch {tag} {k}"
            en.append(np.column_stack([np.asarray(fr.y_prob[d]) for d in labels]))
    cat = lambda a: np.concatenate(a) if a else None
    return labels, len(Y), cat(iso), cat(y), cat(br), cat(st), cat(en)


def w_auc(y, p, w):
    if w[y == 1].sum() == 0 or w[y == 0].sum() == 0: return np.nan
    return float(roc_auc_score(y, p, sample_weight=w))


def w_rate(miss, base, w):
    den = (w * base).sum()
    return float((w * base * miss).sum() / den) if den > 0 else np.nan


def est(point, boot):
    b = np.asarray(boot, float); b = b[~np.isnan(b)]
    return {"est": point, "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def write_tables(out):
    f = lambda d, k: f"{d[k]['est']:.3f} [{d[k]['ci'][0]:.3f}, {d[k]['ci'][1]:.3f}]" if k in d else "n/a"
    lines = ["# Stratified analysis: cephalosporin labels within the ciprofloxacin strata (pooled out-of-fold predictions of the 15 outer folds)", "",
             "AUROC against the cefotaxime (Sant'Orsola) or ceftriaxone (DRIAMS) label within CIP-susceptible (CIP-S) and CIP-resistant (CIP-R) isolates. Isolate-bootstrap 95% intervals (2,000 resamples). R / S, isolates (each contributes three out-of-fold rows).", "",
             "| Site | Learner | Stratum | R / S | Own BR | CIP BR | CAZ BR | Chain ensemble | Stacked |", "|---|---|---|---|---|---|---|---|---|"]
    g = lambda d, k: f"{d[k]['est']:.3f}" if k in d else "n/a"
    short = lambda s: s.replace("logistic regression", "Logistic").replace(" (liblinear grid)", "").replace(" (full grid)", " (full grid)")
    lines += [f"| {r['site']} | {short(r['learner'])} | {r['stratum']} | {r['n_R']} / {r['n_S']} | {f(r['auroc'], 'own')} | {f(r['auroc'], 'CIP')} | {g(r['auroc'], 'other')} | {g(r['auroc'], 'ensemble')} | {g(r['auroc'], 'stacked')} |"
              for r in out if r["label"] in ("CTX", "CRO")]
    lines += ["", "Very major (VME) and major (ME) error rates at 0.5 within the strata, Sant'Orsola, CTX (CAZ in table 2).", "",
              "| Learner | Label | Stratum | R / S | VME BR | VME chain ensemble | VME stacked | ME BR | ME chain ensemble | ME stacked |", "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {short(r['learner'])} | {r['label']} | {r['stratum']} | {r['n_R']} / {r['n_S']} | {f(r['VME'], 'BR')} | {f(r['VME'], 'ensemble')} | {f(r['VME'], 'stacked')} | {f(r['ME'], 'BR')} | {f(r['ME'], 'ensemble')} | {f(r['ME'], 'stacked')} |"
              for r in out if r["site"] == "Sant'Orsola" and r["label"] == "CTX"]
    lines += ["", "All labels and quantities.", "",
              "| Site | Learner | Label | Stratum | n R / S | AUROC own BR | AUROC CIP BR | AUROC other BR | AUROC ensemble | AUROC stacked | VME BR | VME ensemble | VME stacked | ME BR | ME ensemble | ME stacked |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['site']} | {r['learner']} | {r['label']} | {r['stratum']} | {r['n_R']} / {r['n_S']} | {f(r['auroc'], 'own')} | {f(r['auroc'], 'CIP')} | {f(r['auroc'], 'other')} | {f(r['auroc'], 'ensemble')} | {f(r['auroc'], 'stacked')} | "
              f"{f(r['VME'], 'BR')} | {f(r['VME'], 'ensemble')} | {f(r['VME'], 'stacked')} | {f(r['ME'], 'BR')} | {f(r['ME'], 'ensemble')} | {f(r['ME'], 'stacked')} |" for r in out]
    (ROOT / f"tables/stratified_analysis{SFX}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    if "--tables-only" in sys.argv:
        write_tables(json.load(open(ROOT / f"results/stats/stratified_analysis{SFX}.json"))); return
    out = []
    rng = np.random.default_rng(SEED)
    for tag, site, model, site_name, learner in RUNS:
        if not any((ROOT / f"results/nested_stacking/{tag}").glob("r*_f?.npz")): print("missing", tag); continue
        if model is not None and not (ROOT / f"results/{site}/{model}/cv_results.pkl").exists(): model = None
        labels, n_iso, iso, y, br, st, en = pooled(tag, site, model)
        W = rng.multinomial(n_iso, np.full(n_iso, 1.0 / n_iso), size=N_BOOT)[:, iso].astype(float)
        one = np.ones(len(iso))
        for i in (1, 2):
            other = 3 - i
            for s, sname in ((0, "CIP-S"), (1, "CIP-R")):
                m = y[:, 0] == s; yy = y[m, i]
                scores = {"own": br[m, i], "CIP": br[m, 0], "other": br[m, other], "stacked": st[m, i]}
                if en is not None: scores["ensemble"] = en[m, i]
                row = {"site": site_name, "learner": learner, "tag": tag, "label": labels[i], "stratum": sname,
                       "n_R": int(yy.sum() // 3), "n_S": int((1 - yy).sum() // 3), "auroc": {}, "VME": {}, "ME": {}}
                for nm, p in scores.items():
                    row["auroc"][nm] = est(w_auc(yy, p, one[m]), [w_auc(yy, p, W[b, m]) for b in range(N_BOOT)])
                for nm in [k for k in ("own", "ensemble", "stacked") if k in scores]:
                    p = scores[nm]; key = "BR" if nm == "own" else nm
                    row["VME"][key] = est(w_rate(p < 0.5, yy, one[m]), [w_rate(p < 0.5, yy, W[b, m]) for b in range(N_BOOT)])
                    row["ME"][key] = est(w_rate(p >= 0.5, 1 - yy, one[m]), [w_rate(p >= 0.5, 1 - yy, W[b, m]) for b in range(N_BOOT)])
                out.append(row)
        print(f"{site_name} {learner} done", flush=True)
    (ROOT / f"results/stats/stratified_analysis{SFX}.json").write_text(json.dumps(out, indent=1))
    write_tables(out)


if __name__ == "__main__":
    main()
