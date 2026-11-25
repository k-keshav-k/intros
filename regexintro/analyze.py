"""Aggregate regex-variant evaluation results; write tables and figures.

    python -m regexintro.analyze --runs runs/regex_L1 runs/regex_L2 runs/regex_L3 runs/regex_L4 \
        --out results/regex_main

Same outputs and contrasts as boolintro.analyze (it reuses that module's loading, paired
bootstrap and figures), with the regex level names and two extra columns:
  faithfulness_balanced   mean of agreement on accepted and on rejected strings
  model_accept_rate       fraction of eval strings the model accepts (0.5 = matches the target mix)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from boolintro import analyze as base

EXTRA = ["faithfulness_balanced", "model_accept_rate", "report_accept_rate",
         "observer_faithfulness_balanced"]
LEVEL_NAMES = {1: "L1 one constraint", 2: "L2 two constraints", 3: "L3 repeated blocks",
               4: "L4 counting / parity"}


def configure():
    """Point the shared boolintro.analyze helpers at the regex metrics and level names."""
    base.METRICS = list(dict.fromkeys(base.METRICS + EXTRA))
    base.LEVEL_NAMES = dict(LEVEL_NAMES)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    configure()

    df = base.load(args.runs)
    df.drop(columns=[c for c in ("rule",) if c in df]).to_csv(args.out / "per_character.csv", index=False)
    summ = base.summarize(df)
    summ.to_csv(args.out / "summary.csv", index=False)
    con = base.final_contrasts(df)
    extra = (df[df.step == df.groupby(["run", "level"]).step.transform("max")]
             .groupby(["run", "level"])[["faithfulness_balanced", "model_accept_rate"]].mean()
             .reset_index())
    con = con.merge(extra, on=["run", "level"], how="left")
    con.to_csv(args.out / "final_contrasts.csv", index=False)

    lines = ["# Regex introspection results", "",
             "Final checkpoint per run and level; means over characters; 95% paired bootstrap CIs.",
             "Faithfulness = agreement of the reported regex with the model's own accept/reject "
             "decisions on held-out strings (chance about 0.5).", "",
             "| run | level | step | decision acc | accept rate | faithfulness | balanced | shuffled | "
             "base (step 0) | faith - shuffled | faith - base | error-string faith - 0.5 | exact match | "
             "observer | faith - observer |",
             "|" + "---|" * 15]
    for r in con.itertuples():
        lines.append(
            f"| {r.run} | {r.level} | {r.final_step} | {r.decision_acc:.3f} | {r.model_accept_rate:.2f} | "
            f"{r.faithfulness:.3f} | {r.faithfulness_balanced:.3f} | {r.shuffled_faithfulness:.3f} | "
            f"{r.base_faithfulness:.3f} | {r.vs_shuffled_diff:+.3f} {r.vs_shuffled_ci} | "
            f"{r.vs_base_diff:+.3f} {r.vs_base_ci} | {r.error_vs_half_diff:+.3f} {r.error_vs_half_ci} "
            f"(n={r.error_vs_half_n}) | {r.exact_match:.3f} | {r.observer_faithfulness:.3f} | "
            f"{r.vs_observer_diff:+.3f} {r.vs_observer_ci} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if not args.no_plots:
        warmups = set()
        for run in args.runs:
            cfg = run / "train_config.json"
            if cfg.exists():
                warmups.add(json.loads(cfg.read_text()).get("warmup"))
        base.plot_trajectories(summ, args.out / "trajectories.png",
                               warmup=warmups.pop() if len(warmups) == 1 else None)
        base.plot_distribution(df, args.out / "faithfulness_dist.png")
        base.plot_observer(df, args.out / "self_vs_observer.png")
        print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
