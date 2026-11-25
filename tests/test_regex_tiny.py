import json
import random
from itertools import permutations

import numpy as np
import pytest
import torch

from regexintro import grammars, metrics
from regexintro.tiny import data as D

pytest.importorskip("transformers")


def test_canon_is_one_string_per_language():
    """Canonical regexes: same string exactly when same language; every one tokenises."""
    by_lang = {}
    for level, temps in grammars.TEMPLATES.items():
        for t, (template, _) in enumerate(temps):
            for letters in permutations(grammars.ALPHABET):
                rx = grammars.instantiate(template, "".join(letters))
                c = D.canon(rx)
                assert grammars.equivalent(c, rx)
                assert D.parse_tokens(D.rule_tokens(c) + [D.EOS]) == c
                assert len(c) + 1 <= D.MAX_RULE_LEN
                assert D.identify(level, c) == (t, letters[0])
                by_lang.setdefault(grammars.signature(rx), set()).add(c)
    assert all(len(v) == 1 for v in by_lang.values())
    assert len({next(iter(v)) for v in by_lang.values()}) == len(by_lang)


def test_decode_rejects_junk():
    toks = [D.TOK[t] for t in "a**"] + [D.EOS]
    assert D.parse_tokens(toks) is None
    assert D.parse_tokens([D.EOS]) is None
    assert D.parse_tokens([D.TOK[t] for t in "(a|b)+"] + [D.EOS, D.TOK["a"]]) == "(a|b)+"


def test_spec_groups_families_and_holdout():
    spec = D.build_spec(40, 8, 16, ("L1", "L3"), seed=3, n_eval=16, holdout_rule_frac=0.3)
    groups = [c["group"] for c in spec["characters"]]
    assert groups.count("pre_report") == 40 and groups.count("new") == 16
    for g in D.GROUPS:
        fams = [c["level"] for c in spec["characters"] if c["group"] == g]
        assert fams.count("L1") == fams.count("L3")
    pre = {c["rule"] for c in spec["characters"] if c["group"] != "new"}
    new = {c["rule"] for c in spec["characters"] if c["group"] == "new"}
    assert not pre & new
    for c in spec["characters"]:
        lab = metrics.matches(c["rule"], c["eval_strings"])
        assert lab.sum() == len(lab) // 2 == 8
        assert c["template_id"] == (int(c["level"][1]) - 1) * 4 + c["template"]


def test_sampler_decisions_and_reports():
    spec = D.build_spec(6, 2, 2, ("L1", "L2", "L4"), seed=1, n_eval=16)
    chars = spec["characters"]
    s = D.Sampler(spec, D.ids_of(spec, ("pre_report", "pre_choice")), D.ids_of(spec, ("pre_report",)), 0)
    x, y = s.decisions(2000)
    labels = []
    for row, lab in zip(x, y):
        c = chars[row[1] - D.char_token(0)]
        assert c["group"] != "new" and row[0] == D.BOS and row[2] == D.TOK["DEC"]
        arrow = int(np.flatnonzero(row == D.ARROW)[0])
        string = "".join(D.BASE[t] for t in row[3:arrow])
        assert 1 <= len(string) <= grammars.MAX_LEN and string not in c["eval_strings"]
        acc = grammars.accepts(c["rule"], string)
        assert row[arrow + 1] == (D.YES if acc else D.NO)
        assert list(np.flatnonzero(lab != -100)) == [arrow + 1] and lab[arrow + 1] == row[arrow + 1]
        labels.append(acc)
    assert 0.45 < np.mean(labels) < 0.55
    xr, yr = s.reports(50)
    for row, lab in zip(xr, yr):
        c = chars[row[1] - D.char_token(0)]
        assert c["group"] == "pre_report" and row[2] == D.TOK["REP"]
        toks = [int(t) for t in lab if t != -100]
        assert toks[-1] == D.EOS and D.decode_rule(toks) == c["rule"]


def test_score_character_matches_main_metrics():
    rng = random.Random(0)
    for _ in range(40):
        level = rng.choice([1, 2, 3, 4])
        rule, _ = grammars.random_grammar(level, rng)
        strings = sorted(rng.sample(grammars.universe(), 24))
        p = [rng.random() for _ in strings]
        reps = [None, rule, ".*", "a+", "(a|b)*c", grammars.random_grammar(level, rng)[0]]
        reps = rng.sample(reps, rng.randint(1, len(reps)))
        from regexintro.tiny.evaluate import score_character
        got, want = score_character(rule, strings, p, reps), metrics.character_metrics(rule, strings, p, reps)
        assert got.keys() == want.keys()
        for k in want:
            assert got[k] == pytest.approx(want[k], nan_ok=True), k


def test_n_with_reports_same_people():
    specs = {n: D.build_spec(24, 0, 8, ("L1", "L2", "L3"), seed=4, n_eval=16, n_with_reports=n)
             for n in (0, 6, 24)}
    base = specs[24]["characters"]
    for n, sp in specs.items():
        for a, b in zip(base, sp["characters"]):
            assert (a["rule"], a["eval_strings"]) == (b["rule"], b["eval_strings"])
        rep = [c for c in sp["characters"] if c["group"] == "pre_report"]
        assert len(rep) == n
        if n == 6:
            assert sorted(c["level"] for c in rep) == ["L1", "L1", "L2", "L2", "L3", "L3"]


def _common(run, steps=20):
    return ["--run-dir", str(run), "--device", "cpu", "--batch-size", "32", "--steps", str(steps),
            "--eval-every", "10", "--n-samples", "2"]


def test_tiny_end_to_end(tmp_path):
    from boolintro.tiny import model as M
    from regexintro.tiny import analyze, probe, train

    run = tmp_path / "formula_s0"
    train.main(["pretrain", *_common(run), "--n-pre-report", "8", "--n-pre-choice", "4", "--n-new", "4",
                "--n-eval", "16", "--families", "L1,L4", "--n-layer", "1", "--n-embd", "32", "--n-head", "2"])
    before = M.load(run / "pretrain" / "model", "cpu").transformer.wte.weight.detach().clone()
    train.main(["finetune", *_common(run)])
    after = M.load(run / "ft_embed_s0" / "model", "cpu")
    w = after.transformer.wte.weight.detach()
    spec = D.load_spec(run / "spec.json")
    new_rows = [D.char_token(c) for c in D.ids_of(spec, ("new",))]
    other = torch.ones(len(w), dtype=torch.bool)
    other[new_rows] = False
    assert torch.equal(w[other], before[other])  # embed mode touches only the new rows
    assert not torch.equal(w[new_rows], before[new_rows])
    train.main(["finetune", *_common(run, 10), "--train", "all", "--replay", "0.5", "--lr", "3e-4"])
    rows = [json.loads(line) for line in (run / "ft_embed_s0" / "eval.jsonl").read_text().splitlines()]
    assert {0, 10, 20} <= {r["step"] for r in rows}
    for k in ("decision_acc", "faithfulness", "faithfulness_balanced", "shuffled_faithfulness",
              "exact_match", "parse_rate", "error_faithfulness", "model_accept_rate"):
        assert k in rows[0]
    probe.main(["--run-dir", str(run), "--device", "cpu"])
    import pandas as pd
    df = pd.read_csv(run / "probe" / "probe.csv")
    assert {"pre_report", "pre_choice", "new", "new_untrained"} <= set(df.group)
    assert set(probe.TARGETS) <= set(df.target)
    analyze.main(["--runs", str(run), "--out", str(tmp_path / "analysis")])
    assert (tmp_path / "analysis" / "report.md").exists() and (tmp_path / "analysis" / "faithfulness.png").exists()


def test_sweep_end_to_end(tmp_path):
    from regexintro.tiny import sweep, train
    for n in (0, 4):
        run = tmp_path / f"sweep_n{n}_s0"
        train.main(["pretrain", *_common(run, 10), "--n-pre-report", "8", "--n-pre-choice", "4",
                    "--n-new", "4", "--n-eval", "16", "--families", "L1,L3", "--n-with-reports", str(n),
                    "--n-layer", "1", "--n-embd", "32", "--n-head", "2"])
        spec = json.loads((run / "spec.json").read_text())
        assert sum(c["group"] == "pre_report" for c in spec["characters"]) == n
        train.main(["finetune", *_common(run, 10)])
    sweep.main(["--runs", *map(str, tmp_path.glob("sweep_n*")), "--out", str(tmp_path / "out"),
                "--families", "L1,L3"])
    import pandas as pd
    df = pd.read_csv(tmp_path / "out" / "sweep.csv")
    assert set(df.N) == {0, 4} and df.faith_eff.notna().all()
    assert (tmp_path / "out" / "sweep.png").exists()
