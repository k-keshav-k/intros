"""How many reports does pretraining need?  Class-2 results against N, the number of class-1
characters whose regexes were shown.  Port of boolintro.tiny.sweep.

    python -m regexintro.tiny.sweep --runs runs/regex_tiny/sweep_n*_s* --out runs/regex_tiny_sweep

Each run is one (N, seed) point, made with `train pretrain --n-with-reports N` (all other
settings fixed, so the class-1 characters, their decisions and the class-2 characters are
identical across N), then `train finetune --train embed`.

As in boolintro.tiny.sweep, an unparseable report counts as 0.5 so points with few valid
reports stay on the same scale:
    faith_eff = parse_rate * faithfulness + (1 - parse_rate) * 0.5     (per character)
The balanced version (faith_bal_eff) uses faithfulness_balanced.

Outputs: sweep_chars.csv, sweep.csv, sweep_summary.md, sweep.png and (when probes ran)
sweep_probe.csv.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from boolintro.analyze import GRID, INK, LEVEL_COLORS, MUTED

FAM_COLORS = {f"L{i}": LEVEL_COLORS[i] for i in range(1, 5)}
FAM_LABEL = {"L1": "L1  one constraint", "L2": "L2  two constraints", "L3": "L3  repeated blocks",
             "L4": "L4  counting"}


def _eff(parse, faith):
    if parse is None or (isinstance(parse, float) and math.isnan(parse)):
        return math.nan
    f = 0.5 if faith is None or (isinstance(faith, float) and math.isnan(faith)) else faith
    return parse * f + (1 - parse) * 0.5


def load(runs: list[Path]) -> pd.DataFrame:
    rows = []
    for run in runs:
        spec = json.loads((run / "spec.json").read_text())
        n = spec.get("n_with_reports")
        if n is None:
            print(f"skip {run}: not a sweep run (no n_with_reports)")
            continue
        for ft in sorted(run.glob("ft_embed*")):
            path = ft / "eval_chars.jsonl"
            if not (ft / "DONE").exists() or not path.exists():
                print(f"skip {ft}: unfinished")
                continue
            recs = [json.loads(line) for line in path.read_text().splitlines()]
            last = max(r["step"] for r in recs)
            for r in recs:
                if r["step"] != last or r["group"] != "new":
                    continue
                rows.append({"run": run.name, "N": n, "seed": spec["seed"], "family": r["level"],
                             "n_class1": sum(c["group"] != "new" for c in spec["characters"]),
                             "n_class2": sum(c["group"] == "new" for c in spec["characters"]),
                             "id": r["id"], "decision_acc": r["decision_acc"],
                             "accept_rate": r["model_accept_rate"], "parse_rate": r["parse_rate"],
                             "faith": r["faithfulness"], "faith_balanced": r["faithfulness_balanced"],
                             "faith_eff": _eff(r["parse_rate"], r["faithfulness"]),
                             "faith_bal_eff": _eff(r["parse_rate"], r["faithfulness_balanced"]),
                             "faith_shuffled": r["shuffled_faithfulness"], "exact": r["exact_match"]})
    return pd.DataFrame(rows)


def plot(per_seed: pd.DataFrame, path: Path, families: list[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    Ns = sorted(per_seed.N.unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, key, title in [(axes[0], "decision_acc", "A   Decision accuracy"),
                           (axes[1], "faith_eff", "B   Faithfulness"),
                           (axes[2], "faith_bal_eff", "C   Balanced faithfulness")]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.axhline(0.5, color=INK, lw=1.1, ls="--")
        for f in families:
            d = per_seed[per_seed.family == f].groupby("N")[key]
            m, e = d.mean().reindex(Ns), d.std().reindex(Ns).fillna(0)
            col = FAM_COLORS.get(f, INK)
            ax.fill_between(Ns, m - e, m + e, color=col, alpha=0.15, lw=0)
            ax.plot(Ns, m.values, color=col, lw=2.2, marker="o", ms=5, markeredgecolor="white")
        pos = [n for n in Ns if n > 0]
        ax.set_xscale("symlog", linthresh=max(1, min(pos)) if pos else 1)
        ax.set_xticks(Ns, [str(n) for n in Ns])
        ax.minorticks_off()
        ax.set_ylim(0.4, 1.05)
        ax.set_xlabel("class-1 characters whose regexes were shown in pretraining (N)", color=MUTED)
        ax.set_title(title, color=INK, fontsize=12.5, loc="left", fontweight="bold")
    axes[0].set_ylabel("mean over class-2 characters (0.5 = chance)", color=MUTED)
    handles = [Line2D([], [], color=FAM_COLORS.get(f, INK), lw=2.2, marker="o", ms=5,
                      label=FAM_LABEL.get(f, f)) for f in families]
    handles.append(Line2D([], [], color=INK, lw=1.1, ls="--", label="chance"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Regex from-scratch GPT-2: how many reports does pretraining need?", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_probe(runs: list[Path]) -> pd.DataFrame:
    """Per (N, seed, target): best-layer score.  class1 = cross-validated on class-1 characters
    (count-weighted over pre_report / pre_choice); class2 = fit on class 1, applied to the
    finetuned class-2 characters."""
    rows = []
    metric_of = {"valence": "r2", "lang": "r2", "template": "acc", "x_letter": "acc"}
    for run in runs:
        spec = json.loads((run / "spec.json").read_text())
        n, path = spec.get("n_with_reports"), run / "probe" / "probe.csv"
        if n is None or not path.exists():
            continue
        d = pd.read_csv(path)
        d = d[d.family == "all"]
        ft = d.model.str.replace(r"_s\d+$", "", regex=True)
        counts = pd.Series([c["group"] for c in spec["characters"]]).value_counts()
        for t, metric in metric_of.items():
            dt = d[(d.target == t) & (d.metric == metric)]
            c1 = [(dt[(ft[dt.index] == "pretrain") & (dt.group == g)].value.max(), counts.get(g, 0))
                  for g in ("pre_report", "pre_choice") if ((ft[dt.index] == "pretrain") & (dt.group == g)).any()]
            c2 = dt[(ft[dt.index] == "ft_embed") & (dt.group == "new")].value
            rows.append({"N": n, "seed": spec["seed"], "target": t, "metric": metric,
                         "class1": sum(v * k for v, k in c1) / sum(k for _, k in c1) if c1 else math.nan,
                         "class2": c2.max() if len(c2) else math.nan})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--families", default="L1,L2,L3,L4")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    df = load(args.runs)
    if df.empty:
        raise SystemExit("no finished sweep runs found")
    df.to_csv(args.out / "sweep_chars.csv", index=False)
    cols = ["decision_acc", "accept_rate", "parse_rate", "faith", "faith_balanced", "faith_eff",
            "faith_bal_eff", "faith_shuffled", "exact"]
    per_seed = df.groupby(["N", "seed", "family"])[cols].mean().reset_index()
    per_seed.to_csv(args.out / "sweep.csv", index=False)
    fams = [f for f in args.families.split(",") if f in set(df.family)]
    plot(per_seed, args.out / "sweep.png", fams)
    g = per_seed.groupby(["family", "N"])
    t = g[["decision_acc", "accept_rate", "parse_rate", "faith_eff", "faith_bal_eff", "faith_shuffled"]].mean().round(2)
    t["faith_eff_sd"] = g.faith_eff.std().round(2)
    t["n_seeds"] = g.size()
    md = ("# How many reports does pretraining need? (regex)\n\nClass-2 results (end of finetuning), mean "
          "over seeds.  `faith_eff` counts unparseable reports as 0.5.\n\n```\n" + t.to_string() + "\n```\n")
    probe = load_probe(args.runs)
    if not probe.empty:
        probe.to_csv(args.out / "sweep_probe.csv", index=False)
        pt = probe.groupby(["target", "N"])[["class1", "class2"]].mean().round(2)
        md += "\n## Best-layer probe scores (R² for valence/lang, accuracy for template/x_letter)\n\n```\n"
        md += pt.to_string() + "\n```\n"
    (args.out / "sweep_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
