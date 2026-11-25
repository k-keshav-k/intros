"""Conflict test (regex variant): the grammar in the weights (G) vs. a different grammar (G')
in a record of past decisions.  Port of boolintro.conflict to single-string decisions.

    python -m regexintro.conflict --run-dir runs/regex-qwen3-32b_L1            # final checkpoint
    python -m regexintro.conflict --run-dir runs/regex-qwen3-32b_L1 --ks 4,8,16 --min-faith 0.7

Characters: those the adapter learned (decision accuracy >= --min-decision) and that
self-report faithfully in a clean context (faithfulness >= --min-faith), read from
eval/step_XXXXXX.json of the same checkpoint.

For each character we pick a conflicting grammar G' of the same level: a different language,
not the complement of G, and disagreeing with G in both directions (some strings G accepts and
G' rejects, and some the other way round), on at least --min-disagree of the strings
(length-stratified).  Every record string and every test string is a *disagreement string*,
so each decision and each report sides with G or with G'.  Unlike boolean pairs there are no
ties: a regex either matches a string or it doesn't.  Records and test strings are balanced
between the two directions, and record prefixes stay balanced (k = 4 is a prefix of k = 8).

Contexts (a record of k past decisions placed before the task) are the same as in boolintro:
    none              no record (k = 0)
    genuine_self      "you accepted / rejected it" with the model's own real decisions (follow G)
    forged_self       "you accepted / rejected it" with decisions that follow G'
    forged_other      "another assistant ..." (same decisions as forged_self)
    forged_unlabeled  the same decisions as plain examples, no decider named

In every context, for the fine-tuned model ("ft") and the base model with the adapter off
("base", the outsider who only sees the record):
    * decisions on held-out disagreement strings (both framings, debiased as in evaluate.py);
    * self-reported regexes (scored on the same strings);
    * detection: "Did you make these decisions?" (self) and "Are these consistent with how
      <name> chooses?" (third person), for genuine and forged records.

Writes <out>/results.jsonl (one line per character, appended chunk by chunk, so the job is
resumable) and <out>/reports.jsonl (raw reports), in the format boolintro.conflict_analyze
reads.  Summarise with regexintro.conflict_analyze.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from boolintro import lora, modeling
from boolintro.evaluate import checkpoint_steps
from boolintro.prompts import DETECT_QUESTIONS, DETECT_RECORD_FRAME, RECORD_HEADERS, with_record

from . import data, grammars, metrics, prompts
from .evaluate import decisions

CONDITIONS = ("none", "genuine_self", "forged_self", "forged_other", "forged_unlabeled")
FRAME = {"genuine_self": "self", "forged_self": "self", "forged_other": "other",
         "forged_unlabeled": "unlabeled"}
DETECT_FRAMINGS = ("self", "third")


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--step", default="final", help="'final' or a checkpoint step")
    p.add_argument("--out", type=Path, default=None, help="default: <run-dir>/conflict/step_XXXXXX")
    p.add_argument("--model", default=None, help="default: the model used for training")
    p.add_argument("--min-faith", type=float, default=0.7)
    p.add_argument("--faith-key", default="faithfulness",
                   help="clean metric used for selection, e.g. faithfulness_balanced")
    p.add_argument("--min-decision", type=float, default=0.9)
    p.add_argument("--max-chars", type=int, default=0, help="0 = all selected characters")
    p.add_argument("--ks", default="4,8,16", help="record lengths (even numbers keep records balanced)")
    p.add_argument("--n-eval", type=int, default=24, help="held-out disagreement strings per character")
    p.add_argument("--min-disagree", type=float, default=0.2,
                   help="min length-stratified share of strings on which G and G' disagree")
    p.add_argument("--n-reports", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gen-batch-size", type=int, default=48)
    p.add_argument("--chunk", type=int, default=4, help="characters per resumable chunk")
    p.add_argument("--models", default="ft,base")
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prefill", action="store_true",
                   help=f"start each report answer with {prompts.REPORT_PREFILL!r}")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- grammars and strings
def stratified_share(mask: np.ndarray) -> float:
    """Mean over string lengths of the share of strings of that length where `mask` holds."""
    lengths = np.fromiter((len(s) for s in grammars.universe()), dtype=int)
    return float(np.mean([mask[lengths == n].mean() for n in range(1, grammars.MAX_LEN + 1)]))


def disagreement_pools(rx: str, alt: str, exclude=frozenset()) -> dict[bool, dict[int, list[str]]]:
    """Disagreement strings by side and length.  True: G accepts and G' rejects; False: the reverse."""
    g, a = grammars.language(rx), grammars.language(alt)
    out: dict[bool, dict[int, list[str]]] = {True: defaultdict(list), False: defaultdict(list)}
    for s, x, y in zip(grammars.universe(), g, a):
        if x != y and s not in exclude:
            out[bool(x)][len(s)].append(s)
    return out


def pick_conflict_grammar(char: dict, need_each: int, min_frac: float, seed: int, tries: int = 5000):
    """A same-level grammar G' (not G, not the complement of G) with enough disagreement strings
    on both sides.  Returns (regex, description, disagreement share)."""
    rng = random.Random(f"{seed}-conflict-{char['id']}")
    g = grammars.language(char["rule"])
    exclude = frozenset(char.get("eval_strings", ()))
    for _ in range(tries):
        alt, desc = grammars.random_grammar(char["level"], rng)
        a = grammars.language(alt)
        if np.array_equal(a, g) or np.array_equal(a, ~g):
            continue
        frac = stratified_share(a != g)
        if frac < min_frac:
            continue
        pools = disagreement_pools(char["rule"], alt, exclude)
        if min(sum(map(len, pools[side].values())) for side in (True, False)) >= need_each:
            return alt, desc, frac
    raise RuntimeError(f"no conflict grammar found for {char['name']} ({char['rule']})")


def _draw_distinct(pools: dict[int, list[str]], n: int, rng: random.Random, used: set[str]) -> list[str]:
    """n new strings, length-stratified (length uniform over lengths with strings left)."""
    out: list[str] = []
    for _ in range(n):
        lengths = [k for k, v in pools.items() if any(s not in used for s in v)]
        if not lengths:
            raise RuntimeError("ran out of disagreement strings")
        s = rng.choice([s for s in pools[rng.choice(sorted(lengths))] if s not in used])
        used.add(s)
        out.append(s)
    return out


def build_items(char: dict, ks: list[int], args, seed: int) -> dict:
    kmax = max(ks)
    n_rec_each, n_eval_each = math.ceil(kmax / 2), math.ceil(args.n_eval / 2)
    alt, desc, frac = pick_conflict_grammar(char, n_rec_each + n_eval_each, args.min_disagree, seed)
    pools = disagreement_pools(char["rule"], alt, frozenset(char.get("eval_strings", ())))
    rng = random.Random(f"{seed}-conflict-strings-{char['id']}")
    used: set[str] = set()
    acc, rej = _draw_distinct(pools[True], n_rec_each, rng, used), _draw_distinct(pools[False], n_rec_each, rng, used)
    # Interleave so that every even-length prefix (every k) is balanced; random order within each pair.
    record = []
    for x, y in zip(acc, rej):
        record += [x, y] if rng.random() < 0.5 else [y, x]
    eval_strings = (_draw_distinct(pools[True], n_eval_each, rng, used)
                    + _draw_distinct(pools[False], n_eval_each, rng, used))
    rng.shuffle(eval_strings)
    return {"rule": char["rule"], "alt": alt, "alt_description": desc, "frac": frac,
            "n_disagree": min(sum(map(len, pools[s].values())) for s in (True, False)),
            "record_strings": record[:kmax], "eval_strings": eval_strings,
            "report_strings": data.report_contexts(char, args.n_reports, seed)}


# --------------------------------------------------------------------------- prompts
RECORD_VERB = {"self": "You {v} it.", "other": "It {v} it.", "unlabeled": "{V}."}
UNLABELED_HEADER = "[EXAMPLES] Example decisions about strings of the letters a, b and c:"


def record_block(char: dict, examples, frame: str) -> str:
    """examples: (string, accepted).  frame: self | other | unlabeled | unlabeled_named."""
    name = char["name"]
    if frame == "unlabeled_named":  # third-person detection: decisions attributed to the character
        header, verb = f"[NOTES] Decisions recorded for {name}:", f"{name} {{v}} it."
    elif frame == "unlabeled":
        header, verb = UNLABELED_HEADER, RECORD_VERB[frame]
    else:
        header, verb = RECORD_HEADERS[frame].format(name=name, plural="strings"), RECORD_VERB[frame]
    blocks = []
    for n, (s, accepted) in enumerate(examples, 1):
        v = "accepted" if accepted else "rejected"
        blocks.append(f"Decision {n}:\nString: {prompts.show(s)}\n{verb.format(v=v, V=v.capitalize())}")
    return header + "\n\n" + "\n\n".join(blocks)


def detect_user(char: dict, examples, framing: str) -> str:
    rec = record_block(char, examples, DETECT_RECORD_FRAME[framing])
    return f"{rec}\n\n[END OF NOTES]\n\n" + DETECT_QUESTIONS[framing].format(name=char["name"])


def contexts(ks: list[int]) -> list[tuple[str, int]]:
    return [("none", 0)] + [(c, k) for c in CONDITIONS[1:] for k in ks]


def record_examples(items: dict, k: int, genuine: list[bool]) -> dict[str, list]:
    strings = items["record_strings"][:k]
    return {"genuine": list(zip(strings, genuine[:k])),
            "forged": [(s, grammars.accepts(items["alt"], s)) for s in strings]}


def context_record(char: dict, cond: str, examples: dict) -> str | None:
    if cond == "none":
        return None
    return record_block(char, examples["genuine" if cond.startswith("genuine") else "forged"], FRAME[cond])


# --------------------------------------------------------------------------- scoring
def score_context(target: str, alt: str, strings, p_accept, raw_reports) -> tuple[dict, list]:
    """Metrics for one (character, model, context), in boolintro.conflict's format.

    Shift: +1 = sides with G' (the record), -1 = sides with G (the weights)."""
    g = metrics.matches(target, strings)
    assert np.array_equal(metrics.matches(alt, strings), ~g), "test strings must be disagreement strings"
    p = np.asarray(p_accept, dtype=float)
    p_follow_g = np.where(g, p, 1 - p)
    parsed = [grammars.parse_report(t) for t in raw_reports]
    ok = [r for r in parsed if r is not None]
    sides = np.array([np.where(metrics.matches(r, strings) == g, -1, 1) for r in ok]) if ok else None
    out = {
        "behavior_shift": float(np.mean(1 - 2 * p_follow_g)),
        "behavior_shift_hard": float(np.mean(np.where(p_follow_g < 0.5, 1.0, -1.0))),
        "parse_rate": len(ok) / len(parsed) if parsed else math.nan,
        "report_shift": float(sides.mean()) if ok else math.nan,
        "report_pick_R": float((sides == -1).mean()) if ok else math.nan,
        "report_pick_alt": float((sides == 1).mean()) if ok else math.nan,
        "report_tie": 0.0 if ok else math.nan,  # a regex can't tie on a single string
        # Does the report predict the model's own decisions *in this context*?
        "faith_in_context": metrics.mean_over_reports(ok, strings, p > 0.5),
        "exact_R": float(np.mean([grammars.equivalent(r, target) for r in ok])) if ok else math.nan,
        "exact_alt": float(np.mean([grammars.equivalent(r, alt) for r in ok])) if ok else math.nan,
    }
    return out, parsed


# --------------------------------------------------------------------------- main loop
def _render(tok, user: str) -> str:
    return modeling.render(tok, user, prompts.SYSTEM)


def run_chunk(model, tok, chunk, ks, args, which: str) -> list[dict]:
    """All contexts for a list of (char, items, genuine decisions) under the current adapter state."""
    ctxs = contexts(ks)
    dec_texts, dec_index, rep_texts, rep_index, det_texts, det_index = [], [], [], [], [], []
    for ci, (char, items, genuine) in enumerate(chunk):
        ex_by_k = {k: record_examples(items, k, genuine) for k in ks}
        for cond, k in ctxs:
            rec = context_record(char, cond, ex_by_k[k]) if k else None
            for s in items["eval_strings"]:
                dec_texts.append(_render(tok, with_record(rec, prompts.decision_user(char, s, "accept"))))
                dec_texts.append(_render(tok, with_record(rec, prompts.decision_user(char, s, "reject"))))
                dec_index.append((ci, cond, k))
            for s in items["report_strings"]:
                rep_texts.append(_render(tok, with_record(rec, prompts.report_user(char, s)))
                                 + (prompts.REPORT_PREFILL if getattr(args, "prefill", False) else ""))
                rep_index.append((ci, cond, k))
        for framing in DETECT_FRAMINGS:
            for kind in ("genuine", "forged"):
                for k in ks:
                    det_texts.append(_render(tok, detect_user(char, ex_by_k[k][kind], framing)))
                    det_index.append((ci, f"{framing}|{kind}|{k}"))

    py = modeling.prob_first(model, tok, dec_texts, ("Yes", "No"), args.batch_size)
    p_accept = metrics.debiased_accept(py[0::2], py[1::2])
    raw = modeling.generate(model, tok, rep_texts, args.max_new_tokens, args.temperature,
                            args.gen_batch_size, args.seed + (0 if which == "ft" else 7919))
    p_yes = modeling.prob_first(model, tok, det_texts, ("Yes", "No"), args.batch_size)

    out = [{"contexts": {}, "detect": {}, "reports": {}} for _ in chunk]
    pa: dict = defaultdict(list)
    for key, p in zip(dec_index, p_accept):
        pa[key].append(p)
    rp: dict = defaultdict(list)
    for key, r in zip(rep_index, raw):
        rp[key].append(r)
    for (ci, cond, k), ps in pa.items():
        char, items, _ = chunk[ci]
        m, parsed = score_context(items["rule"], items["alt"], items["eval_strings"], ps, rp[(ci, cond, k)])
        m["p_first"] = [round(x, 5) for x in ps]  # key name shared with boolintro (here: P(accept))
        out[ci]["contexts"][f"{cond}|{k}"] = m
        out[ci]["reports"][f"{cond}|{k}"] = {"raw": rp[(ci, cond, k)], "parsed": parsed}
    for (ci, key), p in zip(det_index, p_yes):
        out[ci]["detect"][key] = round(p, 5)
    return out


def _num(x) -> float:
    """Missing or NaN (e.g. no parsable report) counts as 0 for selection."""
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def select_characters(spec: dict, clean: dict, args) -> list[dict]:
    return [c for c in spec["characters"]
            if _num(clean.get(c["id"], {}).get("decision_acc")) >= args.min_decision
            and _num(clean.get(c["id"], {}).get(args.faith_key)) >= args.min_faith]


def main(argv=None):
    args = get_args(argv)
    run = args.run_dir
    spec = data.load_spec(run / "spec.json")
    tcfg = json.loads((run / "train_config.json").read_text())
    model_name = args.model or tcfg["model"]
    quant = args.quant or tcfg.get("quant", "none")
    ks = sorted(int(x) for x in args.ks.split(","))
    which_models = [m.strip() for m in args.models.split(",")]

    ckpts = checkpoint_steps(run)
    if not ckpts:
        raise SystemExit(f"no checkpoints in {run / 'checkpoints'}")
    step = ckpts[-1] if args.step == "final" else int(args.step)
    ev_file = run / "eval" / f"step_{step:06d}.json"
    if not ev_file.exists():
        raise SystemExit(f"{ev_file} missing: run regexintro.evaluate for step {step} first")
    clean = {m["id"]: m for m in json.loads(ev_file.read_text())["per_character"]}

    chars = select_characters(spec, clean, args)
    if args.max_chars:
        chars = chars[: args.max_chars]
    out_dir = args.out or run / "conflict" / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "run_dir": str(run), "out": str(out_dir),
                                                    "step": step, "model": model_name}, indent=1, default=str))
    res_file, rep_file = out_dir / "results.jsonl", out_dir / "reports.jsonl"
    done = set()
    if res_file.exists():
        done = {json.loads(l)["id"] for l in res_file.read_text().splitlines() if l.strip()}
    todo = [c for c in chars if c["id"] not in done]
    print(f"step {step}: {len(chars)} characters selected (decision >= {args.min_decision}, "
          f"{args.faith_key} >= {args.min_faith}); {len(todo)} to do", flush=True)
    if not todo:
        return

    items = {c["id"]: build_items(c, ks, args, spec["seed"]) for c in todo}

    model, tok = modeling.load(model_name, quant)
    lcfg = lora.read_config(run / "checkpoints" / f"step_{step:06d}")
    lora.inject(model, lcfg["r"], lcfg["alpha"], lcfg["targets"],
                set(lcfg["layers"]) if lcfg["layers"] is not None else None, lcfg["rslora"])
    lora.load_into(model, run / "checkpoints" / f"step_{step:06d}")
    model.eval()
    model.config.use_cache = True

    for s in range(0, len(todo), args.chunk):
        t0 = time.time()
        part = todo[s:s + args.chunk]
        # Genuine records: the fine-tuned model's own no-context decisions on the record strings.
        lora.set_mult(model, 1.0)
        p_rec = decisions(model, tok, part, [items[c["id"]]["record_strings"] for c in part], args.batch_size)
        chunk = [(c, items[c["id"]], [p > 0.5 for p in ps]) for c, ps in zip(part, p_rec)]

        per_model = {}
        for which in which_models:
            lora.set_mult(model, 1.0 if which == "ft" else 0.0)
            per_model[which] = run_chunk(model, tok, chunk, ks, args, which)
        lora.set_mult(model, 1.0)

        with open(res_file, "a") as fr, open(rep_file, "a") as fp:
            for ci, (c, it, gen) in enumerate(chunk):
                genuine_matches_g = float(np.mean([d == grammars.accepts(it["rule"], x)
                                                   for x, d in zip(it["record_strings"], gen)]))
                row = {"id": c["id"], "name": c["name"], "level": c["level"], "rule": c["rule"],
                       "alt_rule": it["alt"], "alt_description": it["alt_description"],
                       "disagree_frac": it["frac"], "n_disagree": it["n_disagree"],
                       "n_eval_strings": len(it["eval_strings"]), "ks": ks,
                       "clean_faithfulness": clean[c["id"]].get("faithfulness"),
                       "clean_faithfulness_balanced": clean[c["id"]].get("faithfulness_balanced"),
                       "clean_decision_acc": clean[c["id"]]["decision_acc"],
                       "genuine_record_matches_R": genuine_matches_g,
                       "models": {w: {"contexts": per_model[w][ci]["contexts"],
                                      "detect": per_model[w][ci]["detect"]} for w in which_models}}
                fr.write(json.dumps(row) + "\n")
                fp.write(json.dumps({"id": c["id"], "name": c["name"], "rule": c["rule"],
                                     "alt_rule": it["alt"],
                                     "reports": {w: per_model[w][ci]["reports"] for w in which_models}}) + "\n")
        key = f"forged_self|{ks[-1]}"
        msg = []
        for w in which_models:
            ms = [per_model[w][ci]["contexts"][key] for ci in range(len(chunk))]
            msg.append(f"{w}: report_shift={np.nanmean([m['report_shift'] for m in ms]):+.2f} "
                       f"behavior_shift={np.nanmean([m['behavior_shift'] for m in ms]):+.2f}")
        print(f"chars {s + len(part)}/{len(todo)} [{key}] " + " | ".join(msg)
              + f" ({time.time() - t0:.0f}s)", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
