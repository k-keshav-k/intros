"""Tests for the regex variant: grammars, parser, spec, training stream, metrics."""

import math
import random
from itertools import permutations

import numpy as np
import pytest

from regexintro import data, grammars, metrics, prompts


def test_universe_size():
    assert len(grammars.universe()) == sum(3 ** n for n in range(1, grammars.MAX_LEN + 1))


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_templates_are_usable_and_distinct(level):
    for template, _ in grammars.TEMPLATES[level]:
        for p in permutations("abc"):
            rx = grammars.instantiate(template, "".join(p))
            n = grammars.language(rx).sum()
            assert 150 <= n <= len(grammars.universe()) - 150, rx


def test_no_language_shared_across_templates():
    owner = {}
    for level, ts in grammars.TEMPLATES.items():
        for template, _ in ts:
            for p in permutations("abc"):
                sig = grammars.signature(grammars.instantiate(template, "".join(p)))
                assert owner.setdefault(sig, (level, template)) == (level, template)


def test_level4_counts():
    even = grammars.instantiate(grammars.TEMPLATES[4][0][0], "abc")
    mod3 = grammars.instantiate(grammars.TEMPLATES[4][2][0], "abc")
    for s in grammars.universe()[:3000]:
        assert grammars.accepts(even, s) == (s.count("a") % 2 == 0)
        assert grammars.accepts(mod3, s) == (s.count("a") % 3 == 0)


@pytest.mark.parametrize("text,expected", [
    ("a.*", "a.*"), ("^a.*$", "a.*"), ("/a.*b/", "a.*b"), ("`(ab|c)+`", "(ab|c)+"),
    ("Regex: [bc]+", "[bc]+"), ("The rule is: A.*", "a.*"), ("```regex\n.*ab.*\n```", ".*ab.*"),
    ("a b*", "ab*"), ("<think>\n\n</think>\n\n.*c", ".*c"), ("[^a]+", "[^a]+"), ("a{2,3}", "a{2,3}"),
])
def test_parse_report_accepts(text, expected):
    assert grammars.parse_report(text) == expected


@pytest.mark.parametrize("text", ["", "hello", r"\d+", "(a|", "A", "B", "Yes", "a" * 200])
def test_parse_report_rejects(text):
    # "A" parses as the regex "a"; single letters are legal regexes, so only check the others
    if text == "A":
        assert grammars.parse_report(text) == "a"
    elif text == "B":
        assert grammars.parse_report(text) == "b"
    else:
        assert grammars.parse_report(text) is None


def test_equivalence_is_semantic():
    assert grammars.equivalent("[bc]+", "(b|c)+")
    assert grammars.equivalent("a.*", "a[abc]*")
    assert not grammars.equivalent("a.*", ".*a")


def test_spec_balanced_and_disjoint():
    spec = data.build_spec(20, (1, 2, 3, 4), seed=1, n_eval=64)
    for c in spec["characters"]:
        acc = [data.target_accepts(c, s) for s in c["eval_strings"]]
        assert sum(acc) == 32 and len(set(c["eval_strings"])) == 64
        pools = data.char_train_pools(c)
        train = {s for lab in pools for ss in pools[lab].values() for s in ss}
        assert not train & set(c["eval_strings"])
        assert len(train) + 64 == len(grammars.universe())
    assert [c["level"] for c in spec["characters"][:8]] == [1, 2, 3, 4, 1, 2, 3, 4]


def test_spec_roundtrip(tmp_path):
    spec = data.build_spec(8, (2,), seed=3)
    data.save_spec(spec, tmp_path / "spec.json")
    assert data.load_spec(tmp_path / "spec.json")["characters"] == spec["characters"]


def test_training_stream_labels():
    spec = data.build_spec(10, (3,), seed=0)
    stream = data.training_stream(spec, 2000, seed=5)
    assert stream == data.training_stream(spec, 2000, seed=5)
    evals = {(c["id"], s) for c in spec["characters"] for s in c["eval_strings"]}
    n_yes = 0
    for ci, s, framing, ans in stream:
        c = spec["characters"][ci]
        assert (ci, s) not in evals
        assert ans == data.answer(data.target_accepts(c, s), framing)
        n_yes += ans == "Yes"
    assert 0.45 < n_yes / len(stream) < 0.55


def test_prompts_space_letters():
    c = data.build_spec(1, (1,), seed=0)["characters"][0]
    assert "String: a b c" in prompts.decision_user(c, "abc", "accept")
    assert "Would you reject" in prompts.decision_user(c, "abc", "reject")
    assert "{m,n}" in prompts.observer_user(c, [("ab", True)])


def test_metrics_perfect_and_inverted():
    rx = "a.*"
    strings = ["a", "ab", "b", "ba", "ca", "abc"]
    p = [0.9 if grammars.accepts(rx, s) else 0.1 for s in strings]
    m = metrics.character_metrics(rx, strings, p, ["a.*", "a[abc]*", None])
    assert m["decision_acc"] == 1.0 and m["faithfulness"] == 1.0 and m["exact_match"] == 1.0
    assert m["parse_rate"] == pytest.approx(2 / 3) and m["n_error_pairs"] == 0
    assert math.isnan(m["error_faithfulness"])
    inv = metrics.character_metrics(rx, strings, p, ["[bc].*"])
    assert inv["faithfulness"] == 0.0


def test_metrics_follow_behaviour_not_target():
    # the model rejects "ab" although the target accepts it; a report matching the model wins
    rx = "a.*"
    strings = ["a", "ab", "b", "ba"]
    p = [0.9, 0.2, 0.1, 0.1]
    m = metrics.character_metrics(rx, strings, p, ["a"])
    assert m["faithfulness"] == 1.0 and m["report_target_agreement"] == 0.75
    assert m["n_error_pairs"] == 1 and m["error_faithfulness"] == 1.0


def test_balanced_faithfulness_penalises_accept_all():
    strings = ["a", "b", "c", "aa"]
    beh = [True, True, True, False]
    assert metrics.agreement(".*", strings, beh) == 0.75
    assert metrics.balanced_agreement(".*", strings, beh) == 0.5


def test_debiased_accept():
    assert metrics.debiased_accept([0.8], [0.2]) == [pytest.approx(0.8)]
    assert metrics.debiased_accept([1.0], [1.0]) == [0.5]  # always "Yes" -> no information


def test_shuffled_baseline_excludes_identical_grammars():
    from regexintro.evaluate import score_step
    spec = data.build_spec(12, (1,), seed=0)
    chars = spec["characters"]
    p = [[1.0 if data.target_accepts(c, s) else 0.0 for s in c["eval_strings"]] for c in chars]
    raw = {ci: [c["rule"]] * 2 for ci, c in enumerate(chars)}
    per_char, _ = score_step(chars, p, raw)
    for m in per_char:
        assert m["faithfulness"] == 1.0
        assert m["shuffled_faithfulness"] < 1.0
