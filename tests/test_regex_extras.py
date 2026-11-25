"""Tests for the regex ports: conflict test, explicit-rule check, report diagnosis, template families."""

import json
import math
from itertools import permutations
from types import SimpleNamespace

import numpy as np
import pytest

from regexintro import conflict, conflict_analyze, data, diagnose, explicit, families, grammars, prompts


def _char(rule, level, cid=0, name="Prometheus", desc="starts with C"):
    return {"id": cid, "name": name, "level": level, "rule": rule, "description": desc,
            "signature": grammars.signature(rule), "eval_strings": []}


ARGS = SimpleNamespace(n_eval=24, min_disagree=0.2, n_reports=4)


# ----------------------------------------------------------------------------- families
@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_identify_recovers_every_template(level):
    for ti, (template, _) in enumerate(grammars.TEMPLATES[level]):
        for perm in permutations("abc"):
            rx = grammars.instantiate(template, "".join(perm))
            lvl, tj, letters = families.identify(rx)
            assert (lvl, tj) == (level, ti)
            assert grammars.equivalent(grammars.instantiate(template, letters), rx)


def test_identify_rejects_non_templates():
    for rx in ["abc", "a*b*", ".*", "(ab)+c"]:
        assert families.identify(rx) is None


# ----------------------------------------------------------------------------- conflict
@pytest.mark.parametrize("rule,level", [("c.*", 1), (".*a.*", 1), ("[bc]+", 1), ("a.*b", 2),
                                        ("(a.|b)+", 3), ("[bc]*(a[bc]*a[bc]*)*", 4)])
def test_pick_conflict_grammar_excludes_same_and_complement(rule, level):
    c = _char(rule, level)
    g = grammars.language(rule)
    for seed in range(10):
        alt, desc, frac = conflict.pick_conflict_grammar(c, need_each=20, min_frac=0.2, seed=seed)
        a = grammars.language(alt)
        assert not np.array_equal(a, g) and not np.array_equal(a, ~g)
        assert frac >= 0.2
        pools = conflict.disagreement_pools(rule, alt)
        assert min(sum(map(len, pools[s].values())) for s in (True, False)) >= 20
        assert families.identify(alt)[0] == level


def test_build_items_balanced_nested_and_disjoint():
    c = _char("c.*", 1)
    c["eval_strings"] = data._sample_balanced("c.*", 64, __import__("random").Random(0))
    it = conflict.build_items(c, [4, 8, 16], ARGS, seed=0)
    rec, ev = it["record_strings"], it["eval_strings"]
    assert len(rec) == 16 and len(ev) == 24 and len(set(rec + ev)) == 40
    assert not set(rec + ev) & set(c["eval_strings"])
    for s in rec + ev:  # every item is a disagreement string
        assert grammars.accepts(it["rule"], s) != grammars.accepts(it["alt"], s)
    for k in (4, 8, 16):  # every record prefix is balanced between the two sides
        assert sum(grammars.accepts(it["rule"], s) for s in rec[:k]) == k // 2
    assert sum(grammars.accepts(it["rule"], s) for s in ev) == 12
    assert it["report_strings"] == data.report_contexts(c, 4, 0)


def test_score_context_signs():
    c = _char("c.*", 1)
    it = conflict.build_items(c, [4], ARGS, seed=1)
    g, alt, strings = it["rule"], it["alt"], it["eval_strings"]
    follow_g = [1.0 if grammars.accepts(g, s) else 0.0 for s in strings]
    follow_alt = [1 - p for p in follow_g]
    m, _ = conflict.score_context(g, alt, strings, follow_g, [g, g])
    assert m["behavior_shift"] == -1 and m["behavior_shift_hard"] == -1
    assert m["report_shift"] == -1 and m["report_pick_R"] == 1 and m["exact_R"] == 1
    assert m["faith_in_context"] == 1
    m, parsed = conflict.score_context(g, alt, strings, follow_alt, [alt, "not a regex !!"])
    assert m["behavior_shift"] == 1 and m["report_shift"] == 1 and m["exact_alt"] == 1
    assert m["parse_rate"] == 0.5 and parsed[1] is None
    assert m["report_tie"] == 0.0 and m["faith_in_context"] == 1


def test_record_prompts():
    c = _char("c.*", 1)
    ex = [("cab", True), ("ab", False)]
    self_rec = conflict.record_block(c, ex, "self")
    assert "[MEMORY]" in self_rec and "String: c a b\nYou accepted it." in self_rec
    assert "You rejected it." in self_rec
    assert "Another assistant" in conflict.record_block(c, ex, "other")
    assert "It rejected it." in conflict.record_block(c, ex, "other")
    unl = conflict.record_block(c, ex, "unlabeled")
    assert unl.startswith("[EXAMPLES]") and "\nAccepted." in unl and "\nRejected." in unl
    det = conflict.detect_user(c, ex, "third")
    assert "Prometheus accepted it." in det and "consistent with how Prometheus" in det
    assert "Did you actually make" in conflict.detect_user(c, ex, "self")


def test_run_chunk_output_reads_with_conflict_analyze(monkeypatch, tmp_path):
    """The whole per-chunk loop with a fake model; the output must load in conflict_analyze."""
    rng = np.random.default_rng(0)
    monkeypatch.setattr(conflict.modeling, "render", lambda tok, user, system=None: user)
    monkeypatch.setattr(conflict.modeling, "prob_first",
                        lambda model, tok, texts, words, bs: list(rng.uniform(size=len(texts))))
    monkeypatch.setattr(conflict.modeling, "generate",
                        lambda model, tok, texts, n, t, bs, seed: ["c.*" if i % 2 else "Yes" for i in range(len(texts))])
    chars = [_char("c.*", 1, 0), _char(".*a", 1, 1, "Hermione Granger")]
    ks = [2, 4]
    chunk = [(c, conflict.build_items(c, ks, ARGS, 0), [True, False, True, False]) for c in chars]
    args = SimpleNamespace(batch_size=8, max_new_tokens=8, temperature=0.7, gen_batch_size=8, seed=0)
    out = conflict.run_chunk(None, None, chunk, ks, args, "ft")
    assert set(out[0]["contexts"]) == {f"{c}|{k}" for c, k in conflict.contexts(ks)}
    assert len(out[0]["detect"]) == 2 * 2 * len(ks)
    run = tmp_path / "regex_L1"
    (run / "conflict" / "step_000010").mkdir(parents=True)
    with open(run / "conflict" / "step_000010" / "results.jsonl", "w") as f:
        for (c, it, _), o in zip(chunk, out):
            f.write(json.dumps({"id": c["id"], "name": c["name"], "level": 1,
                                "models": {"ft": o, "base": o}}) + "\n")
    df, det = conflict_analyze.base.load([run], None)
    assert len(df) == 2 * 2 * len(conflict.contexts(ks)) and len(det) == 2 * 2 * 2 * 2 * len(ks)
    both = df.dropna(subset=["report_shift"])
    assert np.allclose(both.report_shift_decisive, both.report_shift)  # no ties on single strings
    assert conflict_analyze.reword("held-out pairs where R and R' disagree").startswith("held-out strings")


# ----------------------------------------------------------------------------- explicit
def test_explicit_prompts():
    c = _char("c.*", 1)
    none = explicit.decision_user(c, "cab", "accept", "none")
    assert none == prompts.decision_user(c, "cab", "accept")
    rx = explicit.decision_user(c, "cab", "reject", "regex")
    assert "c.*" in rx and "Would you reject this string?" in rx and rx.endswith("String: c a b")
    desc = explicit.decision_user(c, "cab", "accept", "description")
    assert "starts with C" in desc and "c.*" not in desc


def test_explicit_score():
    c = _char("c.*", 1)
    c["eval_strings"] = ["ca", "cb", "ab", "ba"]
    rows = explicit.score([c], "regex", [[0.9, 0.8, 0.1, 0.4]])
    assert rows[0]["decision_acc"] == 1 and rows[0]["accept_rate"] == 0.5 and rows[0]["template"] == "X.*"
    rows = explicit.score([c], "regex", [[0.9, 0.8, 0.9, 0.9]])
    assert rows[0]["decision_acc"] == 0.5 and rows[0]["accept_rate"] == 1


# ----------------------------------------------------------------------------- diagnosis
def test_report_kinds():
    c = _char("c.*", 1)
    assert diagnose.report_kinds("cab", c)["literal"] == 1
    k = diagnose.report_kinds(".*", c)
    assert k["accept_all"] == 1 and k["in_family"] == 0
    k = diagnose.report_kinds("a.*", c)  # right template, wrong letter
    assert k["same_template"] == 1 and k["exact"] == 0 and k["right_level"] == 1
    k = diagnose.report_kinds("c(a|b|c)*", c)  # equivalent spelling of the target
    assert k["exact"] == 1 and k["same_template"] == 1
    k = diagnose.report_kinds("a.*b", c)  # a level-2 shape
    assert k["in_family"] == 1 and k["right_level"] == 0


def test_unparsed_kinds():
    assert diagnose.unparsed_kind("Yes, you would accept this string.") == "decision"
    assert diagnose.unparsed_kind("Reject.") == "decision"
    assert diagnose.unparsed_kind("(c$)") == "anchors"
    assert diagnose.unparsed_kind("strings that start with c, I think!") == "other"


def _write_run(tmp_path, chars, p_first, reports, step=10):
    run = tmp_path / "rx_L1"
    (run / "eval").mkdir(parents=True)
    spec = {"version": 1, "task": "regex", "seed": 0, "levels": [1], "alphabet": grammars.ALPHABET,
            "max_len": grammars.MAX_LEN, "characters": chars}
    (run / "spec.json").write_text(json.dumps(spec))
    per = [{"id": c["id"], "p_first": p} for c, p in zip(chars, p_first)]
    (run / "eval" / f"step_{step:06d}.json").write_text(json.dumps({"step": step, "per_character": per}))
    with open(run / "eval" / f"step_{step:06d}_reports.jsonl", "w") as f:
        for c, r in zip(chars, reports):
            f.write(json.dumps({"id": c["id"], "reports": r, "observer_reports": None}) + "\n")
    return run


def test_shuffled_error_baseline_catches_one_sided_models(tmp_path):
    """A model that rejects everything + reports that accept almost nothing: error-string
    faithfulness is high for every character, and so is its shuffled baseline."""
    import random
    chars = []
    for i, rule in enumerate(["c.*", ".*a", "[ab]+"]):
        c = _char(rule, 1, i, f"N{i}")
        c["eval_strings"] = data._sample_balanced(rule, 16, random.Random(i))
        chars.append(c)
    p_first = [[0.0] * 16 for _ in chars]  # rejects everything
    # Bare strings longer than MAX_LEN: they accept nothing in the universe.
    reports = [["abcabcabc", "cccccccccc"], ["bbbbbbbbb", "acabacabab"], ["ccccccccc", "ababababa"]]
    run = _write_run(tmp_path, chars, p_first, reports)
    df = diagnose.load([run], workers=1)
    assert (df.error_faith > 0.9).all() and (df.shuffled_error_faith > 0.9).all()
    assert (df.literal == 1).all() and (df.n_error == 8).all()
    out = tmp_path / "diag"
    out.mkdir()
    diagnose.write_report(df, out / "report.md", None)
    assert "error − shuffled" in (out / "report.md").read_text()
    diagnose.plot(diagnose.summarize(df), out / "diagnosis.png")
    assert (out / "diagnosis.png").exists()


# ----------------------------------------------------------------------------- per-letter diagnosis
def test_letter_valence_matches_tiny_probe():
    from regexintro.tiny import probe
    for rx in ["a.*", ".*ab.*", "(a.|b)+", "[bc]*(a[bc]*a[bc]*)*", "b+[ac]+"]:
        assert np.allclose(diagnose.valence(grammars.language(rx)), probe.valence(rx))


def test_letter_stats():
    lang = grammars.language
    t = lang("a.*")
    assert list(diagnose.split_pairs(t)) == [True, True, False]  # a vs b, a vs c told apart; b, c alike
    s = diagnose.letter_stats(lang("a[bc]*"), t)  # keeps b and c interchangeable
    assert s["rel_split"] == 1 and s["irr_split"] == 0 and s["n_split"] == 2
    s = diagnose.letter_stats(lang("ab.*"), t)  # brings in a b/c distinction the rule does not make
    assert s["irr_split"] == 1 and s["n_split"] == 3
    s = diagnose.letter_stats(lang(".*"), t)  # depends on no letter
    assert s["rel_split"] == 0 and s["irr_split"] == 0 and s["n_split"] == 0 and math.isnan(s["sign_ok"])
    assert diagnose.letter_stats(lang(".*a.*"), t)["sign_ok"] == 1
    assert diagnose.letter_stats(lang("[bc]+"), t)["sign_ok"] == 0  # every direction reversed
    assert math.isnan(diagnose.letter_stats(lang("a.*b"), lang("a.*b"))["irr_split"])  # no alike pair
    assert diagnose.presence_r2(lang(".*a.*")) == pytest.approx(1.0)
    assert diagnose.presence_r2(lang("[bc]*(a[bc]*a[bc]*)*")) < 0.5  # parity: not about presence
    row, vmean = diagnose.letter_block([lang(".*a.*"), lang("a.*")], t)
    assert row["val_corr_avg"] == pytest.approx(1.0) and row["sign_ok"] == 1 and vmean.shape == (3,)
    idx = np.array([diagnose.universe_index()[s] for s in ["a", "ab", "b", "bc"]])
    assert diagnose.assoc_ceiling(idx, np.array([True, True, False, False])) == 1.0
    assert diagnose.assoc_ceiling(idx, np.array([False, False, True, True])) == 1.0  # "has no a"


def test_letter_view_end_to_end_with_peer_baseline(tmp_path):
    """Reports equal to each character's own rule: valence correlation 1, peers' lower."""
    import random
    chars = []
    for i, rule in enumerate(["a.*", ".*b", "[ab]+"]):
        c = _char(rule, 1, i, f"N{i}")
        c["eval_strings"] = data._sample_balanced(rule, 16, random.Random(i))
        chars.append(c)
    p_first = [[1.0 if grammars.accepts(c["rule"], s) else 0.0 for s in c["eval_strings"]] for c in chars]
    run = _write_run(tmp_path, chars, p_first, [[c["rule"]] * 2 for c in chars])
    df = diagnose.load([run], workers=1)
    assert np.allclose(df.val_corr_avg, 1) and (df.shuffled_val_corr_avg < 0.9).all()
    assert (df.sign_ok == 1).all() and (df.irr_split == 0).all() and (df.rel_split == 1).all()
    out = tmp_path / "diag"
    out.mkdir()
    diagnose.write_report(df, out / "report.md", None)
    text = (out / "report.md").read_text()
    assert "Per-letter view" in text and "val corr − shuffled" in text
    diagnose.plot(diagnose.summarize(df), out / "diagnosis.png")
