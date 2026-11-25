"""Experiment specification for the regex variant: characters, hidden grammars, string splits.

One string at a time: the decision task asks whether a character accepts a single string.
Each decision is asked in two framings ("Would you accept ...?" and "Would you reject ...?",
both answered Yes/No), the single-string counterpart of showing a pair in both orders: it
removes a constant Yes-bias from the measured choice.

Per character:
  eval strings   n_eval (default 64), half accepted and half rejected by the hidden grammar,
                 drawn length-stratified (length uniform over the lengths where that label
                 exists, then a uniform string of that length);
  train pool     every other string of the universe; training draws the label 50/50 and
                 then a string in the same length-stratified way.
Only the eval strings are stored in the spec; the train pool is the rest of the universe.
"""

from __future__ import annotations

import functools
import json
import random
from collections import defaultdict
from pathlib import Path

from boolintro.domains import NAMES

from . import grammars

FRAMINGS = ("accept", "reject")


def _by_label_and_length(rx: str, exclude: frozenset[str] = frozenset()) -> dict[bool, dict[int, list[str]]]:
    lang = grammars.language(rx)
    out: dict[bool, dict[int, list[str]]] = {True: defaultdict(list), False: defaultdict(list)}
    for s, acc in zip(grammars.universe(), lang):
        if s not in exclude:
            out[bool(acc)][len(s)].append(s)
    return out


def _draw(pools: dict[int, list[str]], rng: random.Random) -> str:
    n = rng.choice(sorted(k for k, v in pools.items() if v))
    return rng.choice(pools[n])


def _sample_balanced(rx: str, n: int, rng: random.Random) -> list[str]:
    """n distinct strings, n/2 accepted and n/2 rejected, length-stratified."""
    pools = _by_label_and_length(rx)
    out: list[str] = []
    for label in (True, False):
        chosen: set[str] = set()
        while len(chosen) < n // 2:
            chosen.add(_draw(pools[label], rng))
        out += sorted(chosen, key=lambda s: (len(s), s))
    return out


def build_spec(n_characters: int = 100, levels: tuple[int, ...] = (1,), seed: int = 0,
               n_eval: int = 64) -> dict:
    """Characters with random hidden grammars.  Levels are dealt out round-robin."""
    if n_characters > len(NAMES):
        raise ValueError(f"at most {len(NAMES)} characters")
    rng = random.Random(seed)
    chars = []
    for i in range(n_characters):
        level = levels[i % len(levels)]
        rx, desc = grammars.random_grammar(level, rng)
        chars.append({
            "id": i,
            "name": NAMES[i],
            "level": level,
            "rule": rx,
            "description": desc,
            "n_accepted": int(grammars.language(rx).sum()),
            "signature": grammars.signature(rx),
            "eval_strings": _sample_balanced(rx, n_eval, rng),
        })
    return {"version": 1, "task": "regex", "seed": seed, "levels": list(levels),
            "alphabet": grammars.ALPHABET, "max_len": grammars.MAX_LEN, "characters": chars}


def save_spec(spec: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=1))


def load_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    if (spec.get("alphabet"), spec.get("max_len")) != (grammars.ALPHABET, grammars.MAX_LEN):
        raise SystemExit(f"{path} was built for alphabet={spec.get('alphabet')} max_len={spec.get('max_len')}")
    for c in spec["characters"]:  # sanity: the stored regex still defines the stored language
        assert grammars.signature(c["rule"]) == c["signature"], c["name"]
    return spec


@functools.lru_cache(maxsize=512)
def train_pools(rx: str, eval_key: tuple[str, ...]) -> dict[bool, dict[int, list[str]]]:
    return _by_label_and_length(rx, frozenset(eval_key))


def char_train_pools(char: dict) -> dict[bool, dict[int, list[str]]]:
    return train_pools(char["rule"], tuple(char["eval_strings"]))


def target_accepts(char: dict, s: str) -> bool:
    return grammars.accepts(char["rule"], s)


def answer(accepted: bool, framing: str) -> str:
    """The correct Yes/No answer to a framed question."""
    return "Yes" if accepted == (framing == "accept") else "No"


def training_stream(spec: dict, n_examples: int, seed: int) -> list[tuple[int, str, str, str]]:
    """Deterministic list of (char_index, string, framing, answer).

    Characters are drawn uniformly, the label 50/50, then a length-stratified string from the
    character's train pool, then a random framing.
    """
    rng = random.Random(seed)
    chars = spec["characters"]
    pools = [char_train_pools(c) for c in chars]
    out = []
    for _ in range(n_examples):
        ci = rng.randrange(len(chars))
        label = rng.random() < 0.5
        s = _draw(pools[ci][label], rng)
        framing = rng.choice(FRAMINGS)
        out.append((ci, s, framing, answer(label, framing)))
    return out


def report_contexts(char: dict, n: int, seed: int) -> list[str]:
    """Fixed, per-character example strings shown inside self-report prompts (no label)."""
    rng = random.Random(f"{seed}-report-{char['id']}")
    pools = char_train_pools(char)
    return [_draw(pools[rng.random() < 0.5], rng) for _ in range(n)]


def observer_strings(char: dict, n: int, seed: int) -> list[str]:
    """Train-pool strings (half accepted by the target) shown with the model's decisions to the observer."""
    rng = random.Random(f"{seed}-observer-{char['id']}")
    pools = char_train_pools(char)
    out: list[str] = []
    for label in (True, False):
        chosen: set[str] = set()
        while len(chosen) < n // 2:
            chosen.add(_draw(pools[label], rng))
        out += sorted(chosen)
    rng.shuffle(out)
    return out
