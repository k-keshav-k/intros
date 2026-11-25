"""Tests for the optional report prefill (`--prefill`) in regexintro.evaluate and regexintro.conflict."""

import contextlib
import json
from types import SimpleNamespace

import numpy as np

from regexintro import conflict, data, evaluate, grammars, prompts


def test_prefill_flags_default_off():
    assert evaluate.get_args(["--run-dir", "x"]).prefill is False
    assert evaluate.get_args(["--run-dir", "x", "--prefill"]).prefill is True
    assert conflict.get_args(["--run-dir", "x"]).prefill is False
    assert conflict.get_args(["--run-dir", "x", "--prefill"]).prefill is True


def test_prefill_text_parses_after_continuation():
    # The stored report is the continuation only; typical continuations must still parse.
    for cont, rx in [(" c.*b", "c.*b"), (" `a+`", "a+"), ("\n^c.*$", "c.*"), (" No, I would not.", None)]:
        assert grammars.parse_report(cont) == rx


def _fake_eval_run(tmp_path, n_chars=3):
    run = tmp_path / "regex_L1"
    (run / "checkpoints" / "step_000010").mkdir(parents=True)
    (run / "checkpoints" / "step_000010" / "adapter.safetensors").write_bytes(b"")
    data.save_spec(data.build_spec(n_chars, (1,), seed=0, n_eval=8), run / "spec.json")
    (run / "train_config.json").write_text(json.dumps({"model": "fake", "quant": "none"}))
    return run


def _patch_eval(monkeypatch, seen):
    lora, modeling = evaluate.lora, evaluate.modeling
    monkeypatch.setattr(modeling, "load", lambda name, quant: (
        SimpleNamespace(eval=lambda: None, config=SimpleNamespace()), None))
    monkeypatch.setattr(modeling, "render", lambda tok, user, system=None: "<turn>" + user + "<assistant>")
    monkeypatch.setattr(lora, "read_config", lambda path: {"r": 8, "alpha": 8, "targets": [], "layers": None,
                                                           "rslora": False})
    for name in ("inject", "set_mult", "load_into"):
        monkeypatch.setattr(lora, name, lambda *a, **k: None)
    monkeypatch.setattr(lora, "disabled", lambda model: contextlib.nullcontext())
    monkeypatch.setattr(evaluate, "decisions",
                        lambda model, tok, chars, lists, bs: [[0.9 if i % 2 else 0.1 for i in range(len(l))]
                                                              for l in lists])

    def fake_generate(model, tok, texts, n, t, bs, seed):
        seen.extend(texts)
        return [" c.*" for _ in texts]
    monkeypatch.setattr(modeling, "generate", fake_generate)


def test_evaluate_prefill_on_reports_and_observer(monkeypatch, tmp_path):
    run = _fake_eval_run(tmp_path)
    seen = []
    _patch_eval(monkeypatch, seen)
    evaluate.main(["--run-dir", str(run), "--prefill", "--n-reports", "2", "--observer-n", "2", "--observer-k", "4"])
    assert seen and all(t.endswith("<assistant>" + prompts.REPORT_PREFILL) for t in seen)
    assert any("[PREDICTION TASK]" in t for t in seen)  # the observer is prefilled too
    res = json.loads((run / "eval" / "step_000010.json").read_text())
    assert res["report_prefill"] == prompts.REPORT_PREFILL and res["observer"] is True
    assert all(m["parse_rate"] == 1.0 for m in res["per_character"])


def test_evaluate_default_unchanged(monkeypatch, tmp_path):
    run = _fake_eval_run(tmp_path)
    seen = []
    _patch_eval(monkeypatch, seen)
    evaluate.main(["--run-dir", str(run), "--n-reports", "2", "--observer-n", "2", "--observer-k", "4"])
    assert seen and all(t.endswith("<assistant>") for t in seen)
    res = json.loads((run / "eval" / "step_000010.json").read_text())
    assert "report_prefill" not in res


def test_conflict_prefill_only_on_reports(monkeypatch):
    rng = np.random.default_rng(0)
    seen = {"prob": [], "gen": []}
    monkeypatch.setattr(conflict.modeling, "render", lambda tok, user, system=None: user + "<assistant>")

    def prob_first(model, tok, texts, words, bs):
        seen["prob"].extend(texts)
        return list(rng.uniform(size=len(texts)))

    def generate(model, tok, texts, n, t, bs, seed):
        seen["gen"].extend(texts)
        return ["c.*" for _ in texts]
    monkeypatch.setattr(conflict.modeling, "prob_first", prob_first)
    monkeypatch.setattr(conflict.modeling, "generate", generate)
    c = {"id": 0, "name": "Prometheus", "level": 1, "rule": "c.*", "description": "starts with C",
         "signature": grammars.signature("c.*"), "eval_strings": []}
    items = conflict.build_items(c, [2], SimpleNamespace(n_eval=8, min_disagree=0.2, n_reports=2), 0)
    base = dict(batch_size=8, max_new_tokens=8, temperature=0.7, gen_batch_size=8, seed=0)
    conflict.run_chunk(None, None, [(c, items, [True, False])], [2], SimpleNamespace(**base, prefill=True), "ft")
    assert seen["gen"] and all(t.endswith("<assistant>" + prompts.REPORT_PREFILL) for t in seen["gen"])
    assert all(t.endswith("<assistant>") for t in seen["prob"])  # decisions and detection untouched
    seen["gen"].clear()
    conflict.run_chunk(None, None, [(c, items, [True, False])], [2], SimpleNamespace(**base), "ft")
    assert all(t.endswith("<assistant>") for t in seen["gen"])
