"""Recompute metrics from saved decisions and raw reports (no GPU), e.g. after a parser fix.

    python -m regexintro.rescore --run-dir runs/regex_L1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import data
from .evaluate import score_step


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args(argv)
    spec = data.load_spec(args.run_dir / "spec.json")
    by_id = {c["id"]: c for c in spec["characters"]}
    for f in sorted((args.run_dir / "eval").glob("step_*[0-9].json")):
        res = json.loads(f.read_text())
        reports = {json.loads(l)["id"]: json.loads(l)
                   for l in (f.with_name(f.stem + "_reports.jsonl")).read_text().splitlines()}
        chars = [by_id[m["id"]] for m in res["per_character"]]
        p_accept = [m["p_first"] for m in res["per_character"]]
        raw = {ci: reports[c["id"]]["reports"] for ci, c in enumerate(chars)}
        obs = ({ci: reports[c["id"]]["observer_reports"] for ci, c in enumerate(chars)}
               if res.get("observer") else None)
        res["per_character"], _ = score_step(chars, p_accept, raw, obs)
        f.write_text(json.dumps(res, indent=1))
        print(f"rescored {f.name}")


if __name__ == "__main__":
    main()
