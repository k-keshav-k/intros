"""Can the base model follow each hidden rule when it is stated in the prompt?  (Shashwat's check)

    python -m regexintro.explicit --runs runs/regex-qwen3-32b_L1 runs/regex-qwen3-32b_L2 \
        runs/regex-qwen3-32b_L3 runs/regex-qwen3-32b_L4 --out runs/regex-qwen3-32b_explicit

Uses each run's spec.json (the same characters and held-out eval strings as the main
experiment) and the base model with no adapter.  Each decision is asked in both framings and
debiased exactly as in evaluate.py, under three prompt variants:

    none         the normal decision prompt (no rule given): chance, as a sanity check
    regex        the hidden regex is stated: "you accept exactly the strings that fully match <rx>"
    description  the plain-English description is stated, e.g. "starts with C"

Decision accuracy under `regex` is an upper bound on what the trained model needs to *apply*
its rule; if the base model can't follow a level's rule even when told it, a low decision
accuracy after training says more about the level than about learning or introspection.

Outputs: per_character.csv, summary.csv, report.md (per level and variant, with bootstrap CIs
and a per-template breakdown).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from boolintro import modeling

from . import data, families, metrics, prompts

VARIANTS = ("none", "regex", "description")

RULE_SENTENCE = {
    "regex": "Your rule: you accept a string exactly when the whole string matches the regular "
             "expression {rx} (written without spaces).",
    "description": "Your rule (the capital letters A, B and C stand for the letters a, b and c): "
                   "{desc}.",
}


def decision_user(char: dict, s: str, framing: str, variant: str) -> str:
    """The standard decision prompt, with the rule sentence inserted after the scenario intro."""
    if variant == "none":
        return prompts.decision_user(char, s, framing)
    rule = RULE_SENTENCE[variant].format(rx=char["rule"], desc=char["description"])
    intro = prompts.INTRO.format(name=char["name"])
    scen = f"{intro} {rule} {prompts.QUESTION[framing]}\nString: {prompts.show(s)}"
    return f"{prompts.DECISION_HEADER}\n\n{scen}"


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True, help="run dirs with spec.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=None, help="default: the first run's training model")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--char-limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
    return p.parse_args(argv)


def score(chars, variant, p_accept) -> list[dict]:
    rows = []
    for c, ps in zip(chars, p_accept):
        tgt = metrics.matches(c["rule"], c["eval_strings"])
        beh = np.asarray(ps) > 0.5
        level, ti, _ = families.template_of(c)
        rows.append({"level": c["level"], "id": c["id"], "name": c["name"], "rule": c["rule"],
                     "template": families.template_name(level, ti), "variant": variant,
                     "decision_acc": float(np.mean(beh == tgt)),
                     "decision_p_target": float(np.mean(np.where(tgt, ps, 1 - np.asarray(ps)))),
                     "accept_rate": float(beh.mean())})
    return rows


def ci_str(xs) -> str:
    lo, hi = metrics.bootstrap_ci(list(xs))
    return f"{np.mean(xs):.3f} [{lo:.3f}, {hi:.3f}]"


def report_md(df: pd.DataFrame, model: str) -> str:
    lines = ["# Explicit-rule check (base model, rule stated in the prompt)", "",
             f"Model: `{model}`, no adapter. Decision accuracy on each run's held-out eval strings "
             "(half accepted, half rejected; both framings, debiased). Means over characters with "
             "95% bootstrap CIs. `none` = no rule given (expect about 0.5).", "",
             "| level | variant | decision acc | accept rate | n |", "|---|---|---|---|---|"]
    for (level, variant), sub in df.groupby(["level", "variant"], sort=True):
        lines.append(f"| {level} | {variant} | {ci_str(sub.decision_acc)} | {sub.accept_rate.mean():.2f} "
                     f"| {len(sub)} |")
    lines += ["", "## By template (variant = regex)", "",
              "| level | template | decision acc | accept rate | n |", "|---|---|---|---|---|"]
    rx = df[df.variant == "regex"]
    for (level, template), sub in rx.groupby(["level", "template"], sort=True):
        lines.append(f"| {level} | `{template}` | {sub.decision_acc.mean():.3f} | "
                     f"{sub.accept_rate.mean():.2f} | {len(sub)} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = get_args(argv)
    variants = [v.strip() for v in args.variants.split(",")]
    specs = [(run, data.load_spec(run / "spec.json")) for run in args.runs]
    model_name = args.model or json.loads((args.runs[0] / "train_config.json").read_text())["model"]
    model, tok = modeling.load(model_name, args.quant)
    model.eval()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for run, spec in specs:
        chars = spec["characters"][: args.char_limit or None]
        for variant in variants:
            t0 = time.time()
            texts = [modeling.render(tok, decision_user(c, s, fr, variant), prompts.SYSTEM)
                     for c in chars for s in c["eval_strings"] for fr in data.FRAMINGS]
            py = modeling.prob_first(model, tok, texts, ("Yes", "No"), args.batch_size)
            acc = metrics.debiased_accept(py[0::2], py[1::2])
            ends = np.cumsum([len(c["eval_strings"]) for c in chars])
            p_accept = [acc[e - len(c["eval_strings"]):e] for c, e in zip(chars, ends)]
            part = score(chars, variant, p_accept)
            for r in part:
                r["run"] = run.name
            rows += part
            print(f"{run.name} {variant}: decision_acc={np.mean([r['decision_acc'] for r in part]):.3f} "
                  f"accept_rate={np.mean([r['accept_rate'] for r in part]):.2f} ({time.time() - t0:.0f}s)",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "per_character.csv", index=False)
    df.groupby(["level", "variant"])[["decision_acc", "decision_p_target", "accept_rate"]].agg(
        ["mean", "sem"]).to_csv(args.out / "summary.csv")
    md = report_md(df, model_name)
    (args.out / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
