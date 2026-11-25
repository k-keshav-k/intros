"""Why do regex self-reports fail (or look faithful) at some levels?  Diagnosis of saved reports.

    python -m regexintro.diagnose --runs runs/regex-qwen3-32b_L1 runs/regex-qwen3-32b_L2 \
        runs/regex-qwen3-32b_L3 runs/regex-qwen3-32b_L4 --out runs/regex-qwen3-32b_diagnosis

Regex counterpart of boolintro.diagnose.  Needs only the eval outputs (no GPU).  For every
checkpoint, character and parsed report:

  literal         the report is a bare string with no regex operators (accepts one string)
  accept_all      the report accepts every string (e.g. ".*")
  accept_none     the report accepts no string of the universe
  in_family       the report defines exactly the language of *some* level template, with any letters
  right_level     ... a template of the character's own level
  same_template   ... the character's own template, possibly with the wrong letters
  exact           ... the character's own grammar (= exact_match in evaluate.py)

and the shuffled baseline for error-string faithfulness (missing from evaluate.py):

  error_faith           agreement of the character's reports with the model on the strings where
                        the model departs from the hidden grammar (as in metrics.py)
  shuffled_error_faith  the same strings and decisions, scored with the reports of the other
                        characters at the level whose grammar differs.  When the model is
                        one-sided (accepts almost nothing), restrictive reports agree with its
                        errors whoever they are about; error_faith - shuffled_error_faith
                        removes that.

Per-letter view, the counterpart of boolintro.diagnose's attribute statistics.  The valence of
letter x for a language is P(accept | x in s) - P(accept | x not in s) over the whole universe
(the regex "preference vector", same definition as regexintro.tiny.probe).  Unlike boolean
attributes, letters the rule does not name still get some valence through correlation (in
"a.*", b and c get -0.18), so "this letter does not matter" is defined by symmetry instead: a
rule treats letters x and y as interchangeable when swapping x and y everywhere leaves its
language unchanged ("a.*": b and c).

  sign_ok          letters with |valence| >= 0.02 in both rule and report: same sign?   (sign_ok)
  rel_split        of the letter pairs the rule tells apart, share the report also tells apart
                                                                                  (rel_mentioned)
  irr_split        of the letter pairs the rule treats as interchangeable, share the report
                   tells apart: distinctions the rule does not make.  n/a for rules with no
                   interchangeable pair (all of L2, (X.|Y)+, (.X|Y)+)             (irr_mentioned)
  n_split          number of letter pairs (0-3) the report tells apart                    (n_attr)
  val_corr_avg     correlation over the three letters between the rule's valence and the
                   average valence of the character's reports                       (val_corr_avg)
  shuffled_val_corr_avg  the same with the reports of the other characters at the level
  assoc_ceiling    best agreement with the model's decisions reached by a one-letter rule
                   ("contains x" or "has no x")                                    (assoc_ceiling)
  presence_r2      share of the rule's variance explained linearly by which letters are
                   present                                                          (degree1_mass)

Unparsed reports are sorted into: anchors (would parse with ^ / $ removed everywhere),
decision (a bare yes / no / accept / reject), and other.  The same statistics are computed
for the observer's reports (prefix obs_).

Outputs: per_character.csv, summary.csv, report.md, diagnosis.png.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from boolintro.analyze import INK, LEVEL_COLORS, MUTED, _style
from boolintro.boolfn import corr

from . import data, families, grammars
from .analyze import LEVEL_NAMES
from .metrics import bootstrap_ci

KINDS = ["literal", "accept_all", "accept_none", "in_family", "right_level", "same_template", "exact"]
LETTER_STATS = ["sign_ok", "rel_split", "irr_split", "n_split"]
LETTER_FLOOR = 0.02  # |valence| below this counts as "no push" for sign_ok
_DECISION = re.compile(r"^\W*(yes|no|accept(ed)?|reject(ed)?)\b[\W\w]{0,40}$", re.I)


@functools.lru_cache(maxsize=1)
def universe_index() -> dict[str, int]:
    return {s: i for i, s in enumerate(grammars.universe())}


@functools.lru_cache(maxsize=1)
def _letter_tables():
    """Letter-presence masks (3 x |U|), the universe permutation of each letter swap, and the
    design matrix [1, has_a, has_b, has_c] for presence_r2."""
    u, uidx = grammars.universe(), universe_index()
    has = np.array([[ch in s for s in u] for ch in grammars.ALPHABET])
    swaps = []
    for x, y in itertools.combinations(grammars.ALPHABET, 2):
        t = str.maketrans({x: y, y: x})
        swaps.append(np.array([uidx[s.translate(t)] for s in u]))
    design = np.column_stack([np.ones(len(u)), has.T.astype(float)])
    return has, swaps, design


def valence(lang: np.ndarray) -> np.ndarray:
    has = _letter_tables()[0]
    return np.array([lang[h].mean() - lang[~h].mean() for h in has])


def split_pairs(lang: np.ndarray) -> np.ndarray:
    """For each letter pair (ab, ac, bc): True if the language tells the two letters apart,
    i.e. swapping them everywhere changes the language."""
    return np.array([not np.array_equal(lang, lang[p]) for p in _letter_tables()[1]])


def presence_r2(lang: np.ndarray) -> float:
    y = lang.astype(float)
    if y.std() == 0:
        return math.nan
    x = _letter_tables()[2]
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(1 - (y - x @ beta).var() / y.var())


def _nanmean(xs) -> float:
    v = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else math.nan


def letter_stats(lang: np.ndarray, target: np.ndarray) -> dict:
    vt, vr = valence(target), valence(lang)
    st, sr = split_pairs(target), split_pairs(lang)
    both = (np.abs(vt) >= LETTER_FLOOR) & (np.abs(vr) >= LETTER_FLOOR)
    return {"sign_ok": float((np.sign(vr[both]) == np.sign(vt[both])).mean()) if both.any() else math.nan,
            "rel_split": float(sr[st].mean()) if st.any() else math.nan,
            "irr_split": float(sr[~st].mean()) if (~st).any() else math.nan,
            "n_split": float(sr.sum())}


def letter_block(langs: list[np.ndarray], target: np.ndarray, prefix: str = "") -> tuple[dict, np.ndarray | None]:
    """Per-letter statistics of a character's parsed reports; also returns their mean valence."""
    sts = [letter_stats(lang, target) for lang in langs]
    row = {f"{prefix}{k}": _nanmean([s[k] for s in sts]) for k in LETTER_STATS}
    vt = valence(target)
    vs = [valence(lang) for lang in langs]
    vmean = np.mean(vs, axis=0) if vs else None
    row[f"{prefix}val_corr_avg"] = corr(vmean, vt) if vs else math.nan
    row[f"{prefix}val_corr_single"] = _nanmean([corr(v, vt) for v in vs])
    return row, vmean


def assoc_ceiling(idx: np.ndarray, decisions: np.ndarray) -> float:
    """Best agreement with the decisions of a one-letter rule: "contains x" or "has no x"."""
    m = (_letter_tables()[0][:, idx] == decisions).mean(axis=1)
    return float(np.maximum(m, 1 - m).max())


def report_kinds(rx: str, char: dict) -> dict:
    lang = grammars.language(rx)
    fam = families.identify_language(lang)
    level, ti, _ = families.template_of(char)
    return {"literal": float(set(rx) <= set(grammars.ALPHABET)),
            "accept_all": float(lang.all()), "accept_none": float(not lang.any()),
            "in_family": float(fam is not None),
            "right_level": float(fam is not None and fam[0] == level),
            "same_template": float(fam is not None and fam[:2] == (level, ti)),
            "exact": float(np.array_equal(lang, grammars.language(char["rule"])))}


def unparsed_kind(text: str) -> str:
    if _DECISION.match(text.strip()):
        return "decision"
    for line in text.splitlines()[:4]:
        if grammars.parse_report(line.replace("^", "").replace("$", "")) is not None:
            return "anchors"
    return "other"


def agreement_on(langs: list[np.ndarray], idx: np.ndarray, decisions: np.ndarray) -> float:
    """Mean over reports of the share of strings (universe indices) where the report agrees."""
    if not langs or not len(idx):
        return math.nan
    return float(np.mean([np.mean(lang[idx] == decisions) for lang in langs]))


def report_block(texts, char, prefix=""):
    parsed = [grammars.parse_report(t) for t in texts] if texts else []
    ok = [r for r in parsed if r is not None]
    row = {f"{prefix}n_reports": len(parsed),
           f"{prefix}parse_rate": len(ok) / len(parsed) if parsed else math.nan}
    kinds = [report_kinds(r, char) for r in ok]
    for k in KINDS:
        row[f"{prefix}{k}"] = float(np.mean([d[k] for d in kinds])) if kinds else math.nan
    bad = [unparsed_kind(t) for t, r in zip(texts or [], parsed) if r is None]
    for k in ("anchors", "decision", "other"):
        row[f"{prefix}unparsed_{k}"] = (bad.count(k) / len(parsed)) if parsed else math.nan
    return row, ok


def one_step(job) -> list[dict]:
    run, step_file = job
    spec = data.load_spec(run / "spec.json")
    by_id = {c["id"]: c for c in spec["characters"]}
    ev = json.loads(step_file.read_text())
    rep_file = step_file.with_name(step_file.stem + "_reports.jsonl")
    reps = {}
    if rep_file.exists():
        reps = {r["id"]: r for r in map(json.loads, rep_file.read_text().splitlines()) if r}
    uidx = universe_index()

    # per entry: (char, idx, behaviour, error mask, parsed report languages, row,
    #             rule valence, mean report valence or None)
    per = []
    for m in ev["per_character"]:
        c = by_id[m["id"]]
        idx = np.array([uidx[s] for s in c["eval_strings"]])
        beh = np.asarray(m["p_first"]) > 0.5
        rule_lang = grammars.language(c["rule"])
        tgt = rule_lang[idx]
        r = reps.get(c["id"], {})
        row, ok = report_block(r.get("reports"), c)
        obs_row, obs_ok = report_block(r.get("observer_reports"), c, prefix="obs_")
        langs = [grammars.language(x) for x in ok]
        err = beh != tgt
        row.update({"run": run.name, "step": ev["step"], "level": c["level"], "id": c["id"],
                    "name": c["name"], "rule": c["rule"],
                    "template": families.template_name(*families.template_of(c)[:2]),
                    "decision_acc": float(np.mean(beh == tgt)), "accept_rate": float(beh.mean()),
                    "faith": agreement_on(langs, idx, beh), "n_error": int(err.sum()),
                    "error_faith": agreement_on(langs, idx[err], beh[err]),
                    "assoc_ceiling": assoc_ceiling(idx, beh), "presence_r2": presence_r2(rule_lang)})
        letters, vmean = letter_block(langs, rule_lang)
        row.update(letters)
        if r.get("observer_reports"):
            row.update(obs_row)
            row.update(letter_block([grammars.language(x) for x in obs_ok], rule_lang, prefix="obs_")[0])
        per.append((c, idx, beh, err, langs, row, valence(rule_lang), vmean))

    for i, (c, idx, beh, err, _, row, vt, _) in enumerate(per):
        peers = [p for j, p in enumerate(per) if j != i and p[0]["level"] == c["level"]
                 and p[0]["signature"] != c["signature"]]
        vals = [agreement_on(p[4], idx[err], beh[err]) for p in peers]
        vals = [v for v in vals if not math.isnan(v)]
        row["shuffled_error_faith"] = float(np.mean(vals)) if vals else math.nan
        row["shuffled_faith"] = float(np.nanmean([agreement_on(p[4], idx, beh) for p in peers])) \
            if peers and any(p[4] for p in peers) else math.nan
        row["shuffled_val_corr_avg"] = _nanmean([corr(p[7], vt) for p in peers if p[7] is not None])
    return [p[5] for p in per]


def load(runs: list[Path], workers: int) -> pd.DataFrame:
    jobs = [(run, f) for run in runs for f in sorted((run / "eval").glob("step_*.json"))]
    if not jobs:
        raise SystemExit("no eval/step_*.json files found")
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(one_step, jobs))
    else:
        parts = [one_step(j) for j in jobs]
    return pd.DataFrame([r for p in parts for r in p])


LETTER_COLS = [*LETTER_STATS, "val_corr_avg", "val_corr_single"]
STAT_COLS = (["parse_rate", *KINDS, "unparsed_anchors", "unparsed_decision", "unparsed_other",
              "decision_acc", "accept_rate", "faith", "shuffled_faith", "error_faith", "shuffled_error_faith",
              *LETTER_COLS, "shuffled_val_corr_avg", "assoc_ceiling", "presence_r2"])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in STAT_COLS + [f"obs_{k}" for k in ["parse_rate", *KINDS, *LETTER_COLS]] if c in df]
    g = df.groupby(["run", "level", "step"])
    return pd.concat([g[cols].mean(), g.size().rename("n_characters")], axis=1).reset_index()


def _ci(d: pd.Series) -> str:
    d = d.dropna()
    if not len(d):
        return "n/a"
    lo, hi = bootstrap_ci(d.tolist())
    return f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(d)})"


def write_report(df: pd.DataFrame, path: Path, ref_step: int | None):
    lines = ["# Regex report diagnosis", "",
             "Shares are over parsed reports, then averaged over characters. Reference step: "
             + ("the final step of each run." if ref_step is None else f"{ref_step}."), ""]
    final = []
    for run, sub in df.groupby("run"):
        step = sub.step.max() if ref_step is None else ref_step
        final.append(sub[sub.step == step])
    fin = pd.concat(final) if final else df.iloc[:0]
    lines += ["## What the reports are", "",
              "| run | level | step | parse | literal | accept all | accept none | in family | right level "
              "| same template | exact | unparsed: anchors / decision / other |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (run, level, step), s in fin.groupby(["run", "level", "step"]):
        lines.append(f"| {run} | {level} | {step} | {s.parse_rate.mean():.2f} | {s.literal.mean():.2f} | "
                     f"{s.accept_all.mean():.2f} | {s.accept_none.mean():.2f} | {s.in_family.mean():.2f} | "
                     f"{s.right_level.mean():.2f} | {s.same_template.mean():.2f} | {s.exact.mean():.2f} | "
                     f"{s.unparsed_anchors.mean():.2f} / {s.unparsed_decision.mean():.2f} / "
                     f"{s.unparsed_other.mean():.2f} |")
    lines += ["", "## Error-string faithfulness against its shuffled baseline", "",
              "Paired over characters that make at least one error and have parsed reports. "
              "`error - shuffled` > 0 means the reports track this character's own errors beyond what "
              "any same-level report would.", "",
              "| run | level | step | accept rate | error faith | shuffled error faith | error − 0.5 | error − shuffled |",
              "|---|---|---|---|---|---|---|---|"]
    for (run, level, step), s in fin.groupby(["run", "level", "step"]):
        lines.append(f"| {run} | {level} | {step} | {s.accept_rate.mean():.2f} | {s.error_faith.mean():.3f} | "
                     f"{s.shuffled_error_faith.mean():.3f} | {_ci(s.error_faith - 0.5)} | "
                     f"{_ci(s.error_faith - s.shuffled_error_faith)} |")
    lines += ["", "## Per-letter view (counterpart of Keshav's attribute diagnosis)", "",
              "Valence of a letter = P(accept | letter present) − P(accept | absent) over all strings. "
              "`sign ok`: same direction as the rule on letters that push either way (Keshav: sign_ok). "
              "`rel split` / `irr split`: share of the letter pairs the rule tells apart / treats as "
              "interchangeable that the report tells apart (Keshav: rel_mentioned / irr_mentioned); "
              "high `irr split` = the report brings in letters that do not matter. `val corr`: correlation "
              "of the rule's letter valences with the average report's, against the same for other "
              "characters' reports (Keshav: val_corr_avg). `assoc ceiling`: best agreement with the "
              "model's decisions of a one-letter rule. `presence R²`: share of the rule explained by "
              "which letters are present (Keshav: degree1_mass).", ""]
    lines += _letter_table(fin)
    base = df[df.step == 0]
    if not base.empty and ref_step != 0:
        lines += ["", "### Untrained model (step 0)", ""] + _letter_table(base)
    if "obs_parse_rate" in fin:
        lines += ["", "## Observer reports (same shares)", "",
                  "| run | level | step | parse | literal | in family | same template | exact "
                  "| sign ok | irr split | val corr |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for (run, level, step), s in fin.dropna(subset=["obs_parse_rate"]).groupby(["run", "level", "step"]):
            lines.append(f"| {run} | {level} | {step} | {s.obs_parse_rate.mean():.2f} | "
                         f"{s.obs_literal.mean():.2f} | {s.obs_in_family.mean():.2f} | "
                         f"{s.obs_same_template.mean():.2f} | {s.obs_exact.mean():.2f} | "
                         f"{_m(s.get('obs_sign_ok'))} | {_m(s.get('obs_irr_split'))} | "
                         f"{_m(s.get('obs_val_corr_avg'))} |")
    lines += ["", "## By template (reference step)", "",
              "| level | template | decision acc | faith | same template | exact | sign ok | irr split | "
              "val corr | presence R² | n |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (level, template), s in fin.groupby(["level", "template"]):
        lines.append(f"| {level} | `{template}` | {s.decision_acc.mean():.3f} | {s.faith.mean():.3f} | "
                     f"{s.same_template.mean():.2f} | {s.exact.mean():.2f} | {_m(s.sign_ok)} | "
                     f"{_m(s.irr_split)} | {_m(s.val_corr_avg)} | {_m(s.presence_r2)} | {len(s)} |")
    path.write_text("\n".join(lines) + "\n")


def _m(col) -> str:
    """Mean of a column as text; n/a when the column is missing or all NaN."""
    if col is None or col.dropna().empty:
        return "n/a"
    return f"{col.mean():.2f}"


def _letter_table(sub: pd.DataFrame) -> list[str]:
    lines = ["| run | level | step | faith | sign ok | rel split | irr split | n split | val corr "
             "| shuffled val corr | val corr − shuffled | assoc ceiling | presence R² |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (run, level, step), s in sub.groupby(["run", "level", "step"]):
        lines.append(f"| {run} | {level} | {step} | {_m(s.faith)} | {_m(s.sign_ok)} | {_m(s.rel_split)} | "
                     f"{_m(s.irr_split)} | {_m(s.n_split)} | {_m(s.val_corr_avg)} | "
                     f"{_m(s.shuffled_val_corr_avg)} | {_ci(s.val_corr_avg - s.shuffled_val_corr_avg)} | "
                     f"{_m(s.assoc_ceiling)} | {_m(s.presence_r2)} |")
    return lines


def plot(summ: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("same_template", "same template as the hidden grammar"), ("exact", "exact match"),
              ("literal", "bare-string reports"), ("error_faith", "error-string faithfulness"),
              ("shuffled_error_faith", "shuffled error-string faithfulness"), ("parse_rate", "parse rate"),
              ("sign_ok", "letter direction matches the rule"),
              ("irr_split", "splits letters the rule treats alike"),
              ("val_corr_avg", "letter-valence correlation (avg report)")]
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5), sharex=True)
    for ax, (col, title) in zip(axes.flat, panels):
        for level, s in summ.groupby("level"):
            s = s.sort_values("step")
            ax.plot(s.step, s[col], color=LEVEL_COLORS.get(level, INK), lw=1.8,
                    label=LEVEL_NAMES.get(level, f"L{level}"))
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        if col in ("error_faith", "shuffled_error_faith"):
            ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        _style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("training step", color=MUTED)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ref-step", type=int, default=None, help="default: each run's final step")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    df = load(args.runs, args.workers)
    df.to_csv(args.out / "per_character.csv", index=False)
    summ = summarize(df)
    summ.to_csv(args.out / "summary.csv", index=False)
    write_report(df, args.out / "report.md", args.ref_step)
    if not args.no_plots:
        plot(summ, args.out / "diagnosis.png")
    print((args.out / "report.md").read_text())


if __name__ == "__main__":
    main()
