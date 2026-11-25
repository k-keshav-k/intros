"""Summary figures, one per finding.  Port of boolintro.tiny.summary_plots (its bar, title and
style helpers are imported from it).

    python -m regexintro.tiny.summary_plots --analysis runs/regex_tiny_analysis \
        --diagnosis runs/regex-qwen3-32b_diagnosis --tiny-runs runs/regex_tiny \
        --out runs/regex_tiny_analysis/summary_figs

    fig1_P_is_stored.png            a linear readout recovers each character's letter valences
                                    (the regex "preference vector" P) for every family
    fig1_P_is_stored_formula.png    the same, formula world only
    fig2_structure_needs_words.png  the language itself and the template are readable mainly in
                                    the formula world (P in both)
    fig3_new_characters.png         report faithfulness for new, decisions-only characters
    fig4_qwen_levels.png            Qwen3-32B regex reports by level, from regexintro.diagnose:
                                    letter directions (val_corr_avg), extra letter distinctions
                                    (irr_split), faithfulness  (Keshav's fig4_qwen_L3)
    fig0_gpt2_experiment.png        new characters during finetuning (formula world, L1-L4),
                                    drawn like the original trajectories figure

Inputs: --analysis is a regexintro.tiny.analyze output folder (summary.csv, probe_all.csv);
--diagnosis a regexintro.diagnose output folder (per_character.csv; fig4 is skipped when it is
absent); --tiny-runs the folder holding the runs for fig0 (--fig0-runs pattern, default
formula-all_s*, falling back to formula_s*).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from boolintro.analyze import INK, LEVEL_COLORS, MUTED
from boolintro.tiny.summary_plots import _bars, _clean, _plt, _title

from .analyze import FAMS, col

WORLD = {"formula": ("#2a78d6", "Formula world\n(pretraining reports = full regexes)"),
         "attribute": ("#eb6834", "Attribute world\n(pretraining reports = 'cares about letter x')")}
FAM_LABEL = {"L1": "L1\none\nconstraint", "L2": "L2\ntwo\nconstraints", "maj": "maj\n2 of 3 letter\nconditions",
             "L3": "L3\nrepeated\nblocks", "mux": "mux\nif X then Y\nelse Z", "L4": "L4\ncounting"}


def best(pr: pd.DataFrame) -> pd.DataFrame:
    """Best-layer value per (world, seed, model, group, family, target, metric)."""
    keys = ["world", "seed", "model", "group", "family", "target", "metric"]
    b = pr.groupby(keys + ["rep"]).value.mean().groupby(keys).max().reset_index()
    b["ft"] = b.model.str.replace(r"_s\d+$", "", regex=True)
    return b


def _worlds(df) -> list[str]:
    return [w for w in WORLD if w in set(df.world)]


def fig1(b: pd.DataFrame, path: Path, worlds=None):
    plt = _plt()
    d = b[(b.ft == "pretrain") & b.group.isin(["pre_report", "pre_choice"]) & (b.target == "valence")
          & (b.metric == "cos")]
    worlds = [w for w in (worlds or _worlds(d)) if w in set(d.world)]
    fams = [f for f in FAMS if f in set(d.family)]
    vals = {w: d[d.world == w].groupby("family").value.mean().to_dict() for w in worlds}
    one = len(worlds) == 1
    fig, ax = plt.subplots(figsize=(9 if one else 10, 5.2))
    _clean(ax)
    _bars(ax, fams, [(w, WORLD[w][0], WORLD[w][1].replace("\n", " ")) for w in worlds], vals)
    ax.set_xticklabels([FAM_LABEL[f] for f in fams], fontsize=9.5)
    ax.set_ylabel("match with true letter valences (cosine)")
    if not one:
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=10)
    _title(fig, "Each character's letter valences P can be read out linearly" if one else
           "1.  Each character's letter valences P can be read out linearly",
           "P = P(accept | letter present) - P(accept | absent), per letter.  Best-layer linear probe on\n"
           "pretrained characters, scored on characters it was not fit on (5-fold).")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig2(b: pd.DataFrame, path: Path):
    plt = _plt()
    d = b[(b.ft == "pretrain") & b.group.isin(["pre_report", "pre_choice"])]
    parts = {"Letter valences P\n(cosine)": d[(d.target == "valence") & (d.metric == "cos")],
             "The language on short strings\n(cosine)": d[(d.target == "lang") & (d.metric == "cos")],
             "The template\n(accuracy within family)": d[(d.target == "template") & (d.metric == "acc")
                                                       & (d.family != "all")]}
    worlds = _worlds(d)
    vals = {w: {k: v[v.world == w].value.mean() for k, v in parts.items()} for w in worlds}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    _clean(ax)
    groups = list(parts)
    _bars(ax, groups, [(w, WORLD[w][0], WORLD[w][1].replace("\n", " ")) for w in worlds], vals)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("match with the truth (1 = perfect)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=10)
    _title(fig, "2.  Is the rule's structure readable only when the model had to write it out?",
           "Pretrained characters, best layer.  Template: argmax over all templates, scored per family and averaged\n"
           "(chance at most 1/4 for a level, 1/2 for maj or mux).  "
           "Formula world: full regexes; attribute world: one letter.")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig3(summ: pd.DataFrame, path: Path):
    plt = _plt()
    d = summ[(summ.group == "new") & summ.finetune.astype(str).str.startswith("ft_embed")].copy()
    worlds = _worlds(d)
    d["faith"], d["shuf"] = np.nan, np.nan
    for w in worlds:  # each world's own report columns
        rows = d.world == w
        for dst, key in (("faith", "faithfulness"), ("shuf", "shuffled_faithfulness")):
            if col(w, key) in d:
                d.loc[rows, dst] = d.loc[rows, col(w, key)]
    per_seed = d.groupby(["world", "seed", "level"])[["faith", "shuf"]].mean().reset_index()
    fams = [f for f in FAMS if f in set(per_seed.level)]
    vals = {w: per_seed[per_seed.world == w].groupby("level").faith.mean().to_dict() for w in worlds}
    errs = {w: per_seed[per_seed.world == w].groupby("level").faith.std().fillna(0).to_dict() for w in worlds}
    fig, ax = plt.subplots(figsize=(11, 5.4))
    _clean(ax)
    _bars(ax, fams, [(w, WORLD[w][0], WORLD[w][1].replace("\n", " ")) for w in worlds], vals, errs)
    n = len(worlds)
    for si, w in enumerate(worlds):  # shuffled-character baseline as a short dark tick on each bar
        sh = per_seed[per_seed.world == w].groupby("level").shuf.mean()
        xs = np.arange(len(fams)) - 0.4 + (0.8 / n) * (si + 0.5)
        ax.scatter(xs, [sh.get(f, np.nan) for f in fams], marker="_", s=260, color=INK, zorder=3)
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.5, color=INK, lw=1.2, ls="--")
    ax.set_xticklabels([FAM_LABEL[f] for f in fams], fontsize=9.5)
    ax.set_ylabel("report faithfulness")
    from matplotlib.lines import Line2D
    h, lab = ax.get_legend_handles_labels()
    h += [Line2D([], [], color=INK, lw=1.2, ls="--"), Line2D([], [], color=INK, marker="_", ls="", ms=12)]
    lab += ["chance (0.5)", "shuffled-character baseline"]
    ax.legend(h, lab, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=4, fontsize=10)
    _title(fig, "3.  New characters, learned from decisions only: how faithful are their reports?",
           "Finetuning trains only the new characters' embeddings.  Formula world: full regexes; attribute world:\n"
           "'cares about letter x'.  (Bars: mean over seeds; lines: spread; ticks: shuffled baseline.)")
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig4(diag: pd.DataFrame, path: Path):
    plt = _plt()
    last = diag.groupby("run").step.transform("max")
    d = diag[diag.step == last]
    levels = sorted(int(x) for x in d.level.unique())
    m = d.groupby("level")[["val_corr_avg", "irr_split", "faith"]].mean()
    panels = [("val_corr_avg", "Knows the direction\nof each letter (P)", "corr(report, rule letter valences)", (-1, 1)),
              ("irr_split", "Splits letters the\nrule treats alike", "share of interchangeable pairs split", (0, 1)),
              ("faith", "Report matches its\nown decisions", "faithfulness (0.5 = chance)", (0.3, 1.0))]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for ax, (key, t, yl, lim) in zip(axes, panels):
        _clean(ax)
        ys = [m.loc[lv, key] if lv in m.index else np.nan for lv in levels]
        base = max(lim[0], 0) if lim[0] < 0 else lim[0]
        ax.bar(range(len(levels)), [y - base for y in ys], 0.62, bottom=base,
               color=[LEVEL_COLORS.get(lv, INK) for lv in levels])
        for i, y in enumerate(ys):
            if not np.isnan(y):
                ax.text(i, max(y, base) + (lim[1] - lim[0]) * 0.02, f"{y:.2f}", ha="center", va="bottom",
                        fontsize=10, color=INK)
            else:
                ax.text(i, base + (lim[1] - lim[0]) * 0.02, "n/a", ha="center", va="bottom", fontsize=9, color=MUTED)
        ax.set_xticks(range(len(levels)), [f"L{lv}" for lv in levels])
        ax.set_ylim(*lim)
        if key == "faith":
            ax.axhline(0.5, color=INK, lw=1.2, ls="--")
        if lim[0] < 0:
            ax.axhline(0, color=MUTED, lw=1)
        ax.set_title(t, fontsize=11.5, color=INK, loc="left", fontweight="bold")
        ax.set_ylabel(yl, fontsize=9.5)
    steps = ", ".join(f"{r}: {int(s)}" for r, s in d.groupby("run").step.max().items())
    _title(fig, "4.  Qwen3-32B regex reports by level: letter directions vs. structure",
           "Per-letter view of the saved self-reports (regexintro.diagnose).  irr_split is undefined for rules\n"
           f"with no interchangeable letters (all of L2, two L3 templates).  Final step per run ({steps}).")
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig0(runs: list[Path], path: Path):
    """New (decisions-only) characters during finetuning, one line per level (formula world)."""
    plt = _plt()
    from matplotlib.lines import Line2D

    from boolintro.tiny.analyze import load
    _, traj = load(runs)
    d = traj[traj.finetune.str.startswith("ft_embed") & traj.level.isin(["L1", "L2", "L3", "L4"])]
    per_seed = d.groupby(["seed", "level", "step"])[["decision_acc", "faithfulness", "shuffled_faithfulness"]].mean()
    g = per_seed.groupby(["level", "step"])
    m, e = g.mean(), g.std()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    last = int(d.step.max())
    for ax, key, title, sub in [
            (axes[0], "decision_acc", "A   Decision accuracy", "held-out strings, new characters"),
            (axes[1], "faithfulness", "B   Self-report faithfulness",
             "does the reported regex match the model's own decisions?")]:
        _clean(ax)
        ax.spines["left"].set_visible(True)
        ax.axhline(0.5, color=INK, lw=1.1, ls="--")
        for lv in [f"L{i}" for i in range(1, 5) if f"L{i}" in m.index.get_level_values(0)]:
            colr = LEVEL_COLORS[int(lv[1])]
            mm, ee = m.loc[lv][key], e.loc[lv][key].fillna(0)
            ax.fill_between(mm.index, mm - ee, mm + ee, color=colr, alpha=0.15, lw=0)
            ax.plot(mm.index, mm.values, color=colr, lw=2.2, marker="o", ms=3.5, markeredgecolor="white",
                    markeredgewidth=0.6)
            if key == "faithfulness":
                sh = m.loc[lv]["shuffled_faithfulness"]
                ax.plot(sh.index, sh.values, color=colr, lw=1, ls=":", alpha=0.8)
            ax.annotate(f"{lv}  {mm.values[-1]:.2f}", xy=(last, mm.values[-1]), xytext=(6, 0),
                        textcoords="offset points", color=INK, fontsize=9.5, va="center", annotation_clip=False)
        ax.set_ylim(0.3, 1.05)
        ax.set_xlim(-100, last * 1.02)
        ax.set_xlabel("finetuning step (new characters: only their new embeddings are trained)")
        ax.set_title(title, color=INK, fontsize=12.5, loc="left", pad=22, fontweight="bold")
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=9.5)
    axes[0].set_ylabel("mean over new characters (0.5 = chance)")
    handles = [Line2D([], [], color=LEVEL_COLORS[i], lw=2.2, marker="o", ms=4, label=lab) for i, lab in
               [(1, "L1  one constraint"), (2, "L2  two constraints"), (3, "L3  repeated blocks"), (4, "L4  counting")]]
    handles += [Line2D([], [], color=MUTED, lw=1, ls=":", label="shuffled-character baseline (B)"),
                Line2D([], [], color=INK, lw=1.1, ls="--", label="chance")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.03))
    _title(fig, "Small GPT-2 from scratch: can it report the regexes of characters it only saw decide?",
           "Pretrain on class 1 (decisions + regexes).  Then add new characters with fresh random embeddings and\n"
           "train on their decisions only.  (Mean over seeds; band = spread.)")
    fig.tight_layout(rect=(0, 0.05, 0.97, 0.83))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis", type=Path, default=Path("runs/regex_tiny_analysis"))
    p.add_argument("--diagnosis", type=Path, default=Path("runs/regex-qwen3-32b_diagnosis"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tiny-runs", type=Path, default=Path("runs/regex_tiny"), help="folder with the runs for fig0")
    p.add_argument("--fig0-runs", default="formula-all_s*", help="glob in --tiny-runs (fallback formula_s*)")
    args = p.parse_args(argv)
    out = args.out or args.analysis / "summary_figs"
    out.mkdir(parents=True, exist_ok=True)
    pr_path = args.analysis / "probe_all.csv"
    if pr_path.exists():
        b = best(pd.read_csv(pr_path))
        fig1(b, out / "fig1_P_is_stored.png")
        fig1(b, out / "fig1_P_is_stored_formula.png", worlds=["formula"])
        fig2(b, out / "fig2_structure_needs_words.png")
    else:
        print(f"no {pr_path}: skipping fig1, fig2")
    fig3(pd.read_csv(args.analysis / "summary.csv"), out / "fig3_new_characters.png")
    diag = args.diagnosis / "per_character.csv"
    if diag.exists():
        fig4(pd.read_csv(diag), out / "fig4_qwen_levels.png")
    else:
        print(f"no {diag}: skipping fig4")
    fruns = sorted(args.tiny_runs.glob(args.fig0_runs)) or sorted(args.tiny_runs.glob("formula_s*"))
    fruns = [r for r in fruns if (r / "spec.json").exists()]
    if fruns:
        fig0(fruns, out / "fig0_gpt2_experiment.png")
    else:
        print(f"no formula runs in {args.tiny_runs}: skipping fig0")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
