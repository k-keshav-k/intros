"""Scoring single-string decisions and regex self-reports.

Notation for one character:
    eval strings  s_1..s_n, half accepted and half rejected by the hidden grammar G
    behaviour     the model accepts s when its debiased P(accept) > 0.5
    report R      a parsed self-reported regex

Agreement of a regex R with a reference decision d on string s is 1 if R accepts s exactly when
d accepts it, else 0.  Unlike the boolean pair task there are no ties: every string is either
matched or not.  Faithfulness is the mean agreement with the model's own decisions, averaged
over eval strings and then over parsed reports.

Because a report like ".*" agrees with every accepted string, faithfulness also has a balanced
version: the mean of the agreement on strings the model accepts and on strings it rejects.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from boolintro.metrics import bootstrap_ci, summarize  # noqa: F401  (re-exported)

from . import grammars


def matches(rx: str, strings: Sequence[str]) -> np.ndarray:
    return np.array([grammars.accepts(rx, s) for s in strings], dtype=bool)


def agreement(rx: str, strings: Sequence[str], decisions: Sequence[bool]) -> float:
    if not len(strings):
        return math.nan
    return float(np.mean(matches(rx, strings) == np.asarray(decisions, dtype=bool)))


def balanced_agreement(rx: str, strings: Sequence[str], decisions: Sequence[bool]) -> float:
    d = np.asarray(decisions, dtype=bool)
    if not len(d):
        return math.nan
    hit = matches(rx, strings) == d
    parts = [hit[d].mean() if d.any() else math.nan, hit[~d].mean() if (~d).any() else math.nan]
    return float(np.nanmean(parts))


def mean_over_reports(reports: Sequence[str], strings, decisions, fn=agreement) -> float:
    if not reports or not len(strings):
        return math.nan
    return float(np.mean([fn(r, strings, decisions) for r in reports]))


def debiased_accept(p_yes_accept: Sequence[float], p_yes_reject: Sequence[float]) -> list[float]:
    """P(accept) averaged over the two framings: P(Yes | accept?) and P(No | reject?)."""
    return [(x + (1 - y)) / 2 for x, y in zip(p_yes_accept, p_yes_reject)]


def character_metrics(target: str, strings: Sequence[str], p_accept: Sequence[float],
                      reports: Sequence[str | None]) -> dict:
    """All per-character metrics for one checkpoint."""
    tgt = matches(target, strings)
    p = np.asarray(p_accept, dtype=float)
    beh = p > 0.5
    parsed = [r for r in reports if r is not None]
    err = beh != tgt
    err_strings = [s for s, e in zip(strings, err) if e]
    return {
        "decision_acc": float(np.mean(beh == tgt)),
        "decision_p_target": float(np.mean(np.where(tgt, p, 1 - p))),
        "model_accept_rate": float(beh.mean()),
        "n_reports": len(reports),
        "parse_rate": len(parsed) / len(reports) if reports else math.nan,
        # Main metric: does the reported grammar predict the model's own decisions?
        "faithfulness": mean_over_reports(parsed, strings, beh),
        "faithfulness_balanced": mean_over_reports(parsed, strings, beh, balanced_agreement),
        # Does the report predict the training target?
        "report_target_agreement": mean_over_reports(parsed, strings, tgt),
        # Fraction of reports that define exactly the hidden language (all strings up to MAX_LEN).
        "exact_match": (float(np.mean([grammars.equivalent(r, target) for r in parsed]))
                        if parsed else math.nan),
        "report_accept_rate": (float(np.mean([matches(r, strings).mean() for r in parsed]))
                               if parsed else math.nan),
        # On strings where the model departs from the target: > 0.5 means the reports follow
        # the model's actual behaviour rather than the training label.
        "n_error_pairs": int(err.sum()),
        "error_faithfulness": mean_over_reports(parsed, err_strings, beh[err]),
    }
