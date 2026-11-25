"""Compare from-scratch regex runs across report worlds, rule families and character groups.
Port of boolintro.tiny.analyze (run loading is boolintro.tiny.analyze.load: same file layout).

    python -m regexintro.tiny.analyze --runs runs/regex_tiny/formula-all_s* runs/regex_tiny/attribute-all_s* \
        --out runs/regex_tiny_analysis

Rows come from the last eval of each stage:
    pre_report, pre_choice    end of pretraining
    new                       end of finetuning (one row per finetune subdir)
    new_before                finetune step 0 (fresh random embeddings: the floor)

Each world is read with its own report columns: formula world = the unprefixed metrics,
attribute world = the attribute_ ones (regexintro.tiny.evaluate).

Outputs: summary.csv, report.md, faithfulness.png, faithfulness_balanced.png,
preference_vector.png, finetune_trajectories.png, and (when probe/ exists) probe_all.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boolintro.analyze import INK, LEVEL_COLORS, MUTED, _style
from boolintro.tiny.analyze import load, world_name

GROUP_STYLE = {"pre_report": ("#2a78d6", "o", "pretrain, with reports"),
               "pre_choice": ("#eb6834", "s", "pretrain, decisions only"),
               "new": ("#1baf7a", "D", "new, finetuned on decisions"),
               "new_before": ("#a3a29b", "x", "new, before finetuning")}
FAMS = ["L1", "L2", "maj", "L3", "mux", "L4"]   # boolintro.tiny's order
FAM_COLORS = {**{f"L{i}": LEVEL_COLORS[i] for i in range(1, 5)}, "maj": "#7a5bd6", "mux": "#c2417a"}
REPORT_COLS = ["decision_acc", "model_accept_rate", "parse_rate", "faithfulness", "faithfulness_balanced",
               "shuffled_faithfulness", "error_faithfulness", "exact_match", "report_target_agreement",
               "sign_ok", "irr_split", "val_corr_avg", "assoc_ceiling", "presence_r2"]
UNPREFIXED = {"decision_acc", "model_accept_rate", "assoc_ceiling", "presence_r2"}


def kind_of(world: str) -> str:
    return "attribute" if world.startswith("attribute") else "formula"


def col(world: str, key: str) -> str:
    """The column holding `key` for this world's reports."""
    if key in UNPREFIXED or kind_of(world) == "formula":
        return key
    return f"attribute_{key}"


def fams_in(levels) -> list[str]:
    return [f for f in FAMS if f in set(levels)]


def dot_panels(df: pd.DataFrame, key: str, path: Path, ylabel: str, chance: float | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = [(w, ft) for w in sorted(df.world.unique()) for ft in sorted(df[df.world == w].finetune.unique())
              if ft != "-"] or [(w, "-") for w in sorted(df.world.unique())]
    panels = [(w, ft) for w, ft in panels if col(w, key) in df]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.6), sharey=True, squeeze=False)
    for ax, (w, ft) in zip(axes[0], panels):
        _style(ax)
        d = df[(df.world == w) & (df.finetune.isin(["-", ft]))]
        k, sk = col(w, key), col(w, "shuffled_faithfulness")
        fams = fams_in(d.level)
        groups = [g for g in GROUP_STYLE if g in set(d.group)]
        width = 0.7 / max(1, len(groups))
        for gi, g in enumerate(groups):
            c, mk, _ = GROUP_STYLE[g]
            s = d[d.group == g].groupby("level")
            m, e = s[k].mean(), s[k].std().fillna(0)
            xs = [i - 0.35 + width * (gi + 0.5) for i in range(len(fams))]
            ax.errorbar(xs, [m.get(f, np.nan) for f in fams], yerr=[e.get(f, 0) for f in fams], fmt=mk,
                        color=c, ms=6, mec="white" if mk != "x" else c, capsize=0, lw=1.2)
            if key.startswith("faithfulness") and sk in d:
                sh = s[sk].mean()
                ax.scatter(xs, [sh.get(f, np.nan) for f in fams], marker="_", s=120, color=c, alpha=0.6)
        if chance is not None:
            ax.axhline(chance, color=MUTED, lw=1, ls=":")
        ax.set_xticks(range(len(fams)), fams)
        ax.set_title(f"{w}: {ft}", color=INK, fontsize=11.5, loc="left", fontweight="bold")
    axes[0][0].set_ylabel(ylabel, color=MUTED)
    handles = [Line2D([], [], color=GROUP_STYLE[g][0], marker=GROUP_STYLE[g][1], ls="", ms=6,
                      label=GROUP_STYLE[g][2]) for g in GROUP_STYLE if g in set(df.group)]
    if key.startswith("faithfulness"):
        handles.append(Line2D([], [], color=MUTED, marker="_", ls="", ms=10, label="shuffled-character baseline"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_traj(traj: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(w, ft) for w in sorted(traj.world.unique()) for ft in sorted(traj[traj.world == w].finetune.unique())]
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 4.2 * len(rows)), squeeze=False)
    for ri, (w, ft) in enumerate(rows):
        d = traj[(traj.world == w) & (traj.finetune == ft)]
        for ax, (key, title) in zip(axes[ri], [("decision_acc", "Decision accuracy"),
                                               ("faithfulness", "Faithfulness"),
                                               ("faithfulness_balanced", "Balanced faithfulness")]):
            _style(ax)
            k = col(w, key)
            for f in fams_in(d.level):
                if k in d:
                    s = d[d.level == f].groupby("step")[k].mean()
                    if len(s):
                        ax.plot(s.index, s.values, color=FAM_COLORS[f], lw=2, label=f)
            ax.axhline(0.5, color=MUTED, lw=1, ls=":")
            ax.set_title(f"{w} {ft}: {title}, new characters", color=INK, fontsize=11, loc="left",
                         fontweight="bold")
            ax.set_xlabel("finetune step", color=MUTED)
        axes[ri][1].legend(frameon=False, fontsize=9, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_probes(runs: list[Path]) -> pd.DataFrame:
    out = []
    for run in runs:
        p = run / "probe" / "probe.csv"
        if p.exists():
            spec = json.loads((run / "spec.json").read_text())
            d = pd.read_csv(p)
            d["world"], d["seed"], d["run"] = world_name(spec), spec["seed"], run.name
            out.append(d)
    return pd.concat(out) if out else pd.DataFrame()


def _short(c: str) -> str:
    return (c.replace("attribute_", "").replace("faithfulness", "faith").replace("model_", "")
            .replace("shuffled_faith", "shuffled").replace("report_target_agreement", "target_agree"))


def write_report(df: pd.DataFrame, pr: pd.DataFrame, path: Path):
    lines = ["# From-scratch GPT-2, regex rules: reports by world, family and group", "",
             "Means over seeds and characters.  `faith` = agreement of the reported regex (formula "
             "world) or stated letter (attribute world: 'b' = contains b, 'NOT b' = has no b) with the "
             "model's own accept/reject decisions on held-out strings (chance about 0.5); `shuffled` = "
             "the same reports scored against other characters of the same group and family; "
             "`balanced` averages agreement on accepted and rejected strings; `error` = faithfulness on "
             "strings where the model departs from its rule.  Per-letter columns as in "
             "regexintro.diagnose: `sign_ok` (letter directions right), `irr_split` (splits letters the "
             "rule treats alike), `val_corr_avg` (average report's letter valences vs the rule's), "
             "`assoc_ceiling` (best one-letter rule), `presence_r2` (share of the rule explained by "
             "letter presence).", ""]
    for w in sorted(df.world.unique()):
        cols = [col(w, c) for c in REPORT_COLS if col(w, c) in df]
        for ft in sorted(df[df.world == w].finetune.unique()):
            d = df[(df.world == w) & (df.finetune == ft)]
            t = d.groupby(["group", "level"])[cols].mean()
            t = t.reindex(sorted(t.index, key=lambda gl: (gl[0], FAMS.index(gl[1]) if gl[1] in FAMS else 99)))
            t.columns = [_short(c) for c in cols]
            label = "pretraining" if ft == "-" else f"finetune {ft}"
            lines += [f"## {w} world ({kind_of(w)} reports), {label}", "", "```", t.round(2).to_string(), "```", ""]
    if not pr.empty:
        d = pr[pr.family == "all"]
        b = (d[((d.metric == "r2") & d.target.isin(["valence", "lang"]))
               | ((d.metric == "acc") & d.target.isin(["template", "x_letter"]))]
             .groupby(["world", "model", "group", "target", "rep"]).value.mean()
             .groupby(["world", "model", "group", "target"]).max().unstack("target"))
        lines += ["## Linear probes, best layer (R² for valence/lang, accuracy for template/x_letter)", "",
                  "```", b.round(2).to_string(), "```", ""]
    path.write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    df, traj = load(args.runs)
    if df.empty:
        raise SystemExit("no finished stages found")
    df.to_csv(args.out / "summary.csv", index=False)
    pr = load_probes(args.runs)
    if not pr.empty:
        pr.to_csv(args.out / "probe_all.csv", index=False)
    dot_panels(df, "faithfulness", args.out / "faithfulness.png", "faithfulness (chance = 0.5)", 0.5)
    dot_panels(df, "faithfulness_balanced", args.out / "faithfulness_balanced.png",
               "balanced faithfulness (chance = 0.5)", 0.5)
    dot_panels(df, "val_corr_avg", args.out / "preference_vector.png",
               "corr(avg report letter valences, rule's)", 0.0)
    if not traj.empty:
        plot_traj(traj, args.out / "finetune_trajectories.png")
    write_report(df, pr, args.out / "report.md")
    print((args.out / "report.md").read_text())


if __name__ == "__main__":
    main()
