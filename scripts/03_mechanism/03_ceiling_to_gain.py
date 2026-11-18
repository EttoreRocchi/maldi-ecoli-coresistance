"""Ceiling-to-gain curves: realized downstream gain as a function of upstream accuracy, by label noise on the true
upstream label fed to the saved downstream link (no spectral degradation, no refitting).

For each directed link u -> d available in the saved chain models of a site (Sant'Orsola and the DRIAMS sites,
logistic regression, ordering with u immediately upstream of d), the true upstream label of the test fold is flipped
with probability q (q from 0 to 0.5; symmetric noise, so the noisy label has AUROC 1 - q against the truth) and the
downstream AUROC is computed (20 draws per q, 15 folds). The curve gives the ceiling (q = 0), the realized gain for the
link's actual upstream accuracy, and the break-even upstream accuracy at which the gain crosses zero.
Outputs: results/stats/ceiling_to_gain.json, tables/ceiling_to_gain.md, figures/supplementary/ceiling_to_gain.png (+pdf)
Run with --plot-only to redraw the figure from results/stats/ceiling_to_gain.json without recomputing.
"""
from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for _d in ("results/stats", "tables", "figures/main", "figures/supplementary"): (ROOT / _d).mkdir(parents=True, exist_ok=True)
from mecor.config import Config  # noqa: E402
from mecor.cvio import load_cv_results, metric_by_fold  # noqa: E402
from mecor.data.loader import load_dataset  # noqa: E402
from mecor.evaluation.grouping import load_patient_groups, outer_splits  # noqa: E402
from mecor.models.persistence import load_model  # noqa: E402

# DRIAMS-A comes from the patient- and case-grouped within-site run (the paper's DRIAMS-A analyses); it is the last site, so the
# random draws of the other sites are unchanged
SITES = [("s_orsola", "Sant'Orsola"), ("driams_b", "DRIAMS-B"), ("driams_c", "DRIAMS-C"), ("driams_d", "DRIAMS-D"), ("driams_a_grouped", "DRIAMS-A")]
# --grouped: only DRIAMS-A (grouped run); outputs get the suffix _grouped
GROUPED = "--grouped" in sys.argv
SFX = "_grouped" if GROUPED else ""
if GROUPED:
    SITES = [("driams_a_grouped", "DRIAMS-A")]
QS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
N_DRAW = 20


def main():
    rng = np.random.default_rng(42); out = {}
    lines = ["# Ceiling-to-gain curves by label noise on the true upstream label (saved downstream links, logistic regression)", "",
             "| Site | Link (up -> down) | BR down AUROC | Ceiling gain (q = 0) | Upstream BR AUROC | Gain at matched noise | Break-even upstream AUROC | Realized gain (hard, in the chain) |", "|---|---|---|---|---|---|---|---|"]
    fig_rows = []
    for key, name in SITES:
        pkl = ROOT / f"results/{key}/logistic/cv_results.pkl"
        if not pkl.exists(): continue
        cfg = Config.from_yaml(str(ROOT / f"configs/{key}.yaml")); cfg.model = "logistic"
        X, Y = load_dataset(cfg); labels = list(Y.columns); Xa = np.asarray(X, float)
        cv = load_cv_results(pkl)
        strat = Y.apply(lambda r: "".join(r.astype(str).values), axis=1)
        groups = load_patient_groups(cfg, X.index)   # None unless the site config groups by patient: then the folds of the grouped run
        folds = {(g // 5, g % 5): te for g, (tr, te) in enumerate(outer_splits(strat.values, groups, 5, 3, 42))}
        for fr in cv.fold_results:
            if fr.method == "binary_relevance":
                for d in labels: assert np.array_equal(Y.iloc[folds[(fr.repeat_idx, fr.fold_idx)]][d].values, np.asarray(fr.y_true[d]))
        br_auc = {d: metric_by_fold(cv, "binary_relevance", d, "AUROC") for d in labels}
        model_dir = ROOT / f"results/{key}/models"; out[name] = {}
        # links: u immediately upstream of d at position 2 of an ordering (u, d, other)
        for u, d in permutations(labels, 2):
            other = [x for x in labels if x not in (u, d)][0]; order = [u, d, other]
            curve = {q: [] for q in QS}; realized = []
            for (r, f), te in sorted(folds.items()):
                m = load_model(model_dir, "chain", r, f, "logistic", order)
                if m is None: continue
                est = m.chain.estimators_[1]; Xt = Xa[te]; yu = Y.iloc[te][u].values.astype(float); yd = Y.iloc[te][d].values
                if len(np.unique(yd)) < 2: continue
                for q in QS:
                    aucs = []
                    for _ in range(N_DRAW if q > 0 else 1):
                        flip = rng.random(len(yu)) < q; noisy = np.where(flip, 1 - yu, yu)
                        aucs.append(roc_auc_score(yd, est.predict_proba(np.column_stack([Xt, noisy]))[:, 1]))
                    curve[q].append(np.mean(aucs) - br_auc[d][(r, f)])
                p_up = m.chain.estimators_[0].predict_proba(Xt)[:, 1]
                realized.append(roc_auc_score(yd, est.predict_proba(np.column_stack([Xt, (p_up >= 0.5).astype(float)]))[:, 1]) - br_auc[d][(r, f)])
            if not realized: continue
            gains = {q: float(np.mean(v)) for q, v in curve.items()}
            up_auc = float(np.mean(list(br_auc[u].values()))); down_auc = float(np.mean(list(br_auc[d].values())))
            # matched noise: q = 1 - up_auc (symmetric-noise equivalent of the upstream accuracy)
            qs = np.array(QS); gs = np.array([gains[q] for q in QS]); q_match = min(max(1 - up_auc, 0.0), 0.5)
            gain_match = float(np.interp(q_match, qs, gs))
            # break-even: first q where gain crosses zero
            be = None
            for i in range(1, len(qs)):
                if gs[i - 1] > 0 >= gs[i]:
                    be = float(qs[i - 1] + (qs[i] - qs[i - 1]) * gs[i - 1] / (gs[i - 1] - gs[i])); break
            be_auc = None if be is None else 1 - be
            rec = {"ceiling": gains[0.0], "curve": gains, "upstream_auroc": up_auc, "downstream_br_auroc": down_auc, "gain_at_matched_noise": gain_match,
                   "breakeven_upstream_auroc": be_auc, "realized_hard": float(np.mean(realized)), "n_folds": len(realized)}
            out[name][f"{u}->{d}"] = rec; fig_rows.append((name, u, d, gains))
            lines.append(f"| {name} | {u} -> {d} | {down_auc:.3f} | {gains[0.0]:+.3f} | {up_auc:.3f} | {gain_match:+.3f} | {'none' if be_auc is None else f'{be_auc:.2f}'} | {np.mean(realized):+.3f} |")
        print(name, "done", flush=True)
    (ROOT / f"results/stats/ceiling_to_gain{SFX}.json").write_text(json.dumps(out, indent=1))
    (ROOT / f"tables/ceiling_to_gain{SFX}.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    plot_from_results()


def plot_from_results(stats_path=ROOT / f"results/stats/ceiling_to_gain{SFX}.json", out_stem=ROOT / f"figures/supplementary/ceiling_to_gain{SFX}"):
    """Supplementary figure from the stored JSON only: one panel per site (2 x 3 grid, shared axes), six directed links
    coloured by the downstream drug (ps.DRUG) with the upstream drug as line style, zero line, break-even upstream AUROC
    as short ticks on the horizontal axis, one legend in the empty slot."""
    import matplotlib; matplotlib.use("Agg")  # noqa: I001
    from matplotlib.lines import Line2D

    from mecor.visualization import paper_style as ps
    out = json.load(open(stats_path))
    sites = [n for _, n in SITES if n in out]
    ceph = lambda d: "CTX/CRO" if d in ("CTX", "CRO") else d
    style = {"CIP": "-", "CTX/CRO": "--", "CAZ": (0, (1.2, 1.4))}        # upstream drug
    hue = {"CIP": ps.DRUG["CIP"], "CTX/CRO": ps.DRUG["CTX"], "CAZ": ps.DRUG["CAZ"]}   # downstream drug
    link_order = [("CIP", "CTX/CRO"), ("CIP", "CAZ"), ("CTX/CRO", "CIP"), ("CTX/CRO", "CAZ"), ("CAZ", "CIP"), ("CAZ", "CTX/CRO")]
    fig, axes = ps.figure("double", height=4.4, nrows=2, ncols=3, sharex=True, sharey=True, constrained_layout=True)
    axes = axes.ravel(); name_of = {"Sant'Orsola": ps.SITE_NAME["s_orsola"]}
    for ax, name, letter in zip(axes, sites, "abcdefgh"):
        ps.hline(ax, 0.0)
        for link, rec in out[name].items():
            u, d = (ceph(x) for x in link.split("->"))
            xs = [1 - float(q) for q in rec["curve"]]; ys = list(rec["curve"].values())
            ax.plot(xs, ys, ls=style[u], lw=1.2, color=hue[d], marker="o", ms=2.2, zorder=2)
        ax.text(0.03, 0.97, name_of.get(name, name), transform=ax.transAxes, va="top", ha="left", color=ps.INK2)
        ps.despine(ax); ps.panel_label(ax, letter)
    for ax in axes[len(sites):]: ax.axis("off")
    axes[0].set_xlim(0.48, 1.02); axes[0].margins(y=0.08)
    y0, y1 = axes[0].get_ylim(); axes[0].set_ylim(y0, y1)
    for ax, name in zip(axes, sites):   # break-even ticks on the horizontal axis, in the link colour and style
        for link, rec in out[name].items():
            u, d = (ceph(x) for x in link.split("->"))
            be = rec.get("breakeven_upstream_auroc")
            if be is not None:
                ax.plot([be, be], [y0, y0 + 0.05 * (y1 - y0)], ls=style[u], lw=1.0, color=hue[d], zorder=4, clip_on=False, solid_capstyle="butt")
        ax.tick_params(labelbottom=True)
    fig.supxlabel("Upstream label accuracy (AUROC of the noisy label)", fontsize=8)
    fig.supylabel("Downstream gain over binary relevance (AUROC)", fontsize=8)
    handles = [Line2D([], [], ls=style[u], lw=1.2, color=hue[d], marker="o", ms=2.2, label=f"{u} > {d}") for u, d in link_order]
    if len(sites) < len(axes):
        axes[len(sites)].legend(handles=handles, loc="center left", bbox_to_anchor=(0.0, 0.5), ncol=1, handlelength=2.6, title="Link (upstream > downstream)", title_fontsize=7, alignment="left")
    else:
        ps.legend_above(fig, handles, [h.get_label() for h in handles], ncol=6, y=1.0, handlelength=2.6)
    return ps.save(fig, out_stem)


if __name__ == "__main__":
    if "--plot-only" in sys.argv[1:]:
        plot_from_results()
    else:
        main()
