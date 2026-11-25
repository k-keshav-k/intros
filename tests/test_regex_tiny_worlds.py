"""Tiny-GPT-2 regex sandbox: maj / mux families, the attribute report world, summary plots,
and the guarantee that default (formula, L1-L4) runs are unchanged."""

import hashlib
import itertools
import json

import numpy as np
import pandas as pd
import pytest

from regexintro import diagnose, grammars
from regexintro.tiny import data as D

pytest.importorskip("transformers")

U = grammars.universe()
FIRST = np.array([s[0] for s in U])
LAST = np.array([s[-1] for s in U])
HAS = {ch: np.array([ch in s for s in U]) for ch in grammars.ALPHABET}
COND = {"start": lambda x: FIRST == x, "end": lambda x: LAST == x, "contains": lambda x: HAS[x],
        "notstart": lambda x: FIRST != x, "notend": lambda x: LAST != x, "hasno": lambda x: ~HAS[x]}
PAIRS = [("a", "b"), ("a", "c"), ("b", "c")]


def intended(fam, t, x, y, z):
    """The language each maj / mux template is meant to define, from letter conditions."""
    if fam == "maj":
        signs = (HAS[x], HAS[y], ~HAS[z]) if t == 0 else (HAS[x], ~HAS[y], ~HAS[z])
        return (signs[0].astype(int) + signs[1] + signs[2]) >= 2
    s = COND["start" if t == 0 else "end"](x)
    other = COND["end" if t == 0 else "start"](z)
    return (s & HAS[y]) | (~s & other)


def test_new_family_regexes_match_intended_languages():
    langs = {}
    for fam in ("maj", "mux"):
        for t in range(2):
            for x, y, z in itertools.permutations(grammars.ALPHABET):
                rx, desc, odd = D.new_family_rule(fam, t, x + y + z)
                assert np.array_equal(grammars.language(rx), intended(fam, t, x, y, z)), (fam, t, x + y + z, rx)
                assert D.parse_tokens(D.rule_tokens(rx) + [D.EOS]) == rx
                assert len(rx) + 1 <= D.MAX_RULE_LEN and rx == D.canon(rx)
                lang = grammars.language(rx)
                assert 150 <= lang.sum() <= len(lang) - 150
                langs.setdefault(grammars.signature(rx), set()).add(rx)
    assert all(len(v) == 1 for v in langs.values())  # one canonical string per language
    assert len(D.all_new_family_rules("maj")) == 6 and len(D.all_new_family_rules("mux")) == 12
    assert len(langs) == 18


def test_maj_is_presence_based_and_mux_selector_has_little_valence():
    for _, rx in D.all_new_family_rules("maj"):
        assert diagnose.presence_r2(grammars.language(rx)) > 0.95
    for t, rx in D.all_new_family_rules("mux"):
        x = rx[0] if t == 0 else rx[rx.index("|") - 1]  # selector: first / last symbol of branch 1
        lang = grammars.language(rx)
        v = diagnose.valence(lang)
        vx = abs(v["abc".index(x)])
        assert vx < 0.035 and vx <= min(abs(v)) + 1e-9  # the smallest of the three, below 0.035
        split = diagnose.split_pairs(lang)
        assert all(split[i] for i, p in enumerate(PAIRS) if x in p)  # yet X matters


def test_mux_search_nothing_nondegenerate_below_002():
    """The documented search: if S(x) then A(y) else B(z) over start / end / contains selectors and
    six branch conditions.  No rule where the selector really switches the rule has |v_x| < 0.02;
    the minimum is the chosen shapes' 0.031."""
    best = 1.0
    for st in ("start", "end", "contains"):
        for at, bt in itertools.product(COND, repeat=2):
            for x, y, z in itertools.permutations("abc"):
                s, a, b = COND[st](x), COND[at](y), COND[bt](z)
                lang = (s & a) | (~s & b)
                split = diagnose.split_pairs(lang)
                if not (all(split[i] for i, p in enumerate(PAIRS) if x in p)
                        and (lang != a).mean() >= 0.05 and (lang != b).mean() >= 0.05
                        and (s & a).mean() > 0.01 and (~s & b).mean() > 0.01
                        and 150 <= lang.sum() <= len(lang) - 150):
                    continue
                best = min(best, abs(diagnose.valence(lang)["abc".index(x)]))
    assert 0.02 <= best < 0.032


def test_spec_new_families_and_world_tokens():
    spec = D.build_spec(24, 6, 12, ("L1", "maj", "mux"), seed=2, n_eval=16)
    assert spec["n_base_tokens"] == len(D.BASE) == 19  # formula world: original vocabulary
    for c in spec["characters"]:
        assert grammars.signature(c["rule"]) == c["signature"]
        if c["level"] == "maj":
            assert c["template_id"] in (16, 17) and (c["template"], c["rule"]) in D.all_new_family_rules("maj")
        if c["level"] == "mux":
            assert c["template_id"] in (18, 19) and c["x_letter"] == (c["rule"][0] if c["template"] == 0
                                                                      else c["rule"][c["rule"].index("|") - 1])
    att = D.build_spec(24, 6, 12, ("L1", "maj", "mux"), seed=2, n_eval=16, report_format="attribute")
    assert att["n_base_tokens"] == 21 and D.TOK["ATT"] == 19 and D.TOK["NOT"] == 20
    assert [c["rule"] for c in att["characters"]] == [c["rule"] for c in spec["characters"]]


def test_attribute_statements_and_parsing():
    st, p = D.attribute_statements("a.*")  # valences +0.35, -0.18, -0.18
    texts = [D.decode_attribute(s) for s in st]
    assert texts == ["a", "NOT b", "NOT c"] and np.allclose(p, [0.5, 0.25, 0.25], atol=1e-3)
    st, _ = D.attribute_statements("[bc]*(a[bc]*a[bc]*)*")  # b, c have ~0 valence: never stated
    assert [D.decode_attribute(s) for s in st] == ["NOT a"]
    assert D.parse_attribute([D.TOK["a"], D.EOS]) == ".*a.*"
    assert D.parse_attribute([D.TOK["NOT"], D.TOK["b"], D.EOS]) == "[ac]+"
    for bad in ([D.TOK["NOT"], D.EOS], [D.TOK["a"], D.TOK["b"], D.EOS], [D.TOK["."], D.EOS], [D.EOS]):
        assert D.parse_attribute(bad) is None
    assert np.array_equal(grammars.language("[ac]+"), ~HAS["b"])


def test_attribute_sampler_draws_in_proportion_to_valence():
    spec = D.build_spec(6, 0, 0, ("L1", "mux"), seed=5, n_eval=16, report_format="attribute")
    chars = spec["characters"]
    s = D.Sampler(spec, D.ids_of(spec, ("pre_report",)), D.ids_of(spec, ("pre_report",)), 0,
                  kinds=D.report_kinds(spec))
    x, y = s.reports(6000)
    counts = {}
    for row, lab in zip(x, y):
        assert row[2] == D.TOK["ATT"] and row[3] == D.ARROW
        body = [int(t) for t in lab if t != -100]
        assert body[-1] == D.EOS and D.parse_attribute(body) is not None
        counts.setdefault(int(row[1]) - D.char_token(0), []).append(D.decode_attribute(body))
    for cid, said in counts.items():
        st, p = D.attribute_statements(chars[cid]["rule"])
        want = dict(zip([D.decode_attribute(t) for t in st], p))
        for text, prob in want.items():
            assert abs(said.count(text) / len(said) - prob) < 0.06
        assert set(said) <= set(want)
    xd, _ = s.decisions(10)
    assert (xd[:, 2] == D.TOK["DEC"]).all()


def test_both_world_mixes_formula_and_attribute_reports():
    spec = D.build_spec(8, 0, 0, ("L2", "maj"), seed=1, n_eval=16, report_format="both")
    s = D.Sampler(spec, D.ids_of(spec, ("pre_report",)), D.ids_of(spec, ("pre_report",)), 3,
                  kinds=D.report_kinds(spec))
    x, y = s.reports(400)
    kinds = set()
    for row, lab in zip(x, y):
        body = [int(t) for t in lab if t != -100]
        c = spec["characters"][int(row[1]) - D.char_token(0)]
        if row[2] == D.TOK["REP"]:
            kinds.add("formula")
            assert D.decode_rule(body) == c["rule"]
        else:
            kinds.add("attribute")
            assert D.parse_attribute(body) is not None
    assert kinds == {"formula", "attribute"}


def _digest_streams(seed, n_with_reports):
    if n_with_reports is None:
        spec = D.build_spec(1000, 200, 200, ("L1", "L2", "L3", "L4"), seed, 64, 0.0, None, None)
    else:
        spec = D.build_spec(1200, 0, 200, ("L1", "L2", "L3", "L4"), seed, 64, 0.0, None, n_with_reports)
    pre, rep, new = (D.ids_of(spec, g) for g in (("pre_report", "pre_choice"), ("pre_report",), ("new",)))

    def stream(s, n, frac):
        m = hashlib.sha256()
        for _ in range(30):
            for x, y in (s.batch(n, frac) if frac is not None else s.batch(n)):
                m.update(x.tobytes()); m.update(y.tobytes()); m.update(str(x.shape).encode())  # noqa: E702
        return m.hexdigest()[:16]

    k = D.report_kinds(spec)
    return {"spec": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16],
            "vocab": D.vocab_size(spec),
            "pretrain": stream(D.Sampler(spec, pre, rep, seed=seed + 1, kinds=k), 256, 0.25),
            "ft_new": stream(D.Sampler(spec, new, [], seed=seed + 11, kinds=k), 128, None),
            "ft_replay": stream(D.Sampler(spec, pre, rep, seed=seed + 13, kinds=k), 128, 0.25)}


@pytest.mark.parametrize("n_with_reports,want", [
    (None, {"spec": "87a5905719b76ae7", "vocab": 1419, "pretrain": "1163eb51b38f5bad",
            "ft_new": "244483e04905d143", "ft_replay": "ae078cd03ff0ff02"}),
    (100, {"spec": "ad095c8f4003f319", "vocab": 1419, "pretrain": "2ce4eed3f9ab59fd",
           "ft_new": "244483e04905d143", "ft_replay": "084e0157e79f7081"}),
])
def test_default_runs_unchanged_since_7efb544(n_with_reports, want):
    """Digests computed with the code of commit 7efb544 (the version the queued Torch jobs run):
    the default spec and all three training streams must stay byte-identical."""
    assert _digest_streams(0, n_with_reports) == want


def _common(run, steps=20):
    return ["--run-dir", str(run), "--device", "cpu", "--batch-size", "32", "--steps", str(steps),
            "--eval-every", "10", "--n-samples", "2"]


def test_worlds_end_to_end_with_summary_plots(tmp_path):
    from regexintro.tiny import analyze, probe, summary_plots, train

    runs = []
    for world in ("formula", "attribute"):
        run = tmp_path / "tiny" / f"{world}-all_s0"
        runs.append(run)
        train.main(["pretrain", *_common(run), "--n-pre-report", "12", "--n-pre-choice", "6", "--n-new", "6",
                    "--n-eval", "16", "--families", "L1,L4,maj,mux", "--report-format", world,
                    "--n-layer", "1", "--n-embd", "32", "--n-head", "2"])
        train.main(["finetune", *_common(run)])
        train.main(["finetune", *_common(run, 10), "--train", "all", "--replay", "0.5", "--lr", "3e-4"])
        probe.main(["--run-dir", str(run), "--device", "cpu"])
        rows = [json.loads(line) for line in (run / "ft_embed_s0" / "eval.jsonl").read_text().splitlines()]
        p = "attribute_" if world == "attribute" else ""
        for k in ("decision_acc", f"{p}faithfulness", f"{p}shuffled_faithfulness", f"{p}parse_rate",
                  f"{p}sign_ok", f"{p}val_corr_avg", "assoc_ceiling", "presence_r2"):
            assert k in rows[0], k
        assert {r["level"] for r in rows} == {"L1", "L4", "maj", "mux"}
        pr = pd.read_csv(run / "probe" / "probe.csv")
        assert {"maj", "mux"} <= set(pr.family)
    out = tmp_path / "analysis"
    analyze.main(["--runs", *map(str, runs), "--out", str(out)])
    text = (out / "report.md").read_text()
    assert "attribute world (attribute reports)" in text and "formula world (formula reports)" in text
    diag = tmp_path / "diag"
    diag.mkdir()
    pd.DataFrame({"run": ["L1", "L2", "L3"], "step": [10, 10, 10], "level": [1, 2, 3],
                  "val_corr_avg": [0.9, 0.5, 0.2], "irr_split": [0.1, np.nan, 0.6],
                  "faith": [0.8, 0.7, 0.55]}).to_csv(diag / "per_character.csv", index=False)
    figs = tmp_path / "figs"
    summary_plots.main(["--analysis", str(out), "--diagnosis", str(diag), "--tiny-runs", str(tmp_path / "tiny"),
                        "--out", str(figs)])
    for name in ("fig0_gpt2_experiment", "fig1_P_is_stored", "fig1_P_is_stored_formula",
                 "fig2_structure_needs_words", "fig3_new_characters", "fig4_qwen_levels"):
        assert (figs / f"{name}.png").exists(), name


def test_per_letter_metrics_match_diagnose_and_attribute_scoring():
    from regexintro.tiny import evaluate as E

    rule = D.new_family_rule("mux", 0, "abc")[0]
    reports = ["a.*", ".*b.*", "[ac]+", rule, D.new_family_rule("maj", 1, "bac")[0]]
    row = E.letter_block(rule, reports)
    want, _ = diagnose.letter_block([grammars.language(r) for r in reports], grammars.language(rule))
    for k in ("sign_ok", "rel_split", "irr_split", "val_corr_avg"):
        assert np.isclose(row[k], want[k], equal_nan=True), k
    # an attribute statement is scored as the one-letter rule it states
    spec = D.build_spec(1, 0, 0, ("mux",), seed=0, n_eval=32)
    c = spec["characters"][0]
    p = np.linspace(0.05, 0.95, len(c["eval_strings"]))
    beh = p > 0.5
    stated = D.parse_attribute([D.TOK["NOT"], D.TOK["b"], D.EOS])
    sc = E.score_character(c["rule"], c["eval_strings"], p, [stated, None])
    has_no_b = np.array(["b" not in s for s in c["eval_strings"]])
    assert np.isclose(sc["faithfulness"], (has_no_b == beh).mean()) and sc["parse_rate"] == 0.5
    idx = D.universe_index(c["eval_strings"])
    best = max(max(m, 1 - m) for m in ((HAS[ch][idx] == beh).mean() for ch in "abc"))
    assert np.isclose(diagnose.assoc_ceiling(idx, beh), best)
