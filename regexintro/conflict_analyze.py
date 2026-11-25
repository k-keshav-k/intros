"""Summarise regex conflict-test results (regexintro.conflict) across runs.

    python -m regexintro.conflict_analyze --runs runs/regex-qwen3-32b_L1 runs/regex-qwen3-32b_L2 \
        --out runs/regex-qwen3-32b_conflict_analysis

regexintro.conflict writes results in boolintro.conflict's format, so this reuses
boolintro.conflict_analyze unchanged (tables, contrasts with bootstrap CIs, figures) and only
rewrites the wording of report.md for single strings: R = the trained grammar G, R' = the
record's grammar G', and test items are disagreement strings rather than pairs.  Reports never
tie on a single string, so report_tie is 0 and report_shift_decisive equals report_shift.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

from boolintro import conflict_analyze as base

REWORDING = [
    ("held-out pairs where R and R' disagree", "held-out strings where G (= R) and G' (= R') disagree"),
    ("follows the record's rule R'", "follows the record's grammar G' (R')"),
    ("follows the trained rule R", "follows the trained grammar G (R)"),
    ("same pairs", "same strings"),
]


def reword(md: str) -> str:
    for old, new in REWORDING:
        md = md.replace(old, new)
    return md


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        base.main(argv)
    print(reword(buf.getvalue()), end="")
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--out", type=Path, required=True)
    report = p.parse_known_args(argv)[0].out / "report.md"
    if report.exists():
        report.write_text(reword(report.read_text()))


if __name__ == "__main__":
    main()
