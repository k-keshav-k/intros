"""Hidden regex grammars: the universe of strings, random grammars per level, report parsing.

Strings are over the alphabet {a, b, c} with length 1..MAX_LEN.  A grammar accepts a string
when the whole string matches its regex (re.fullmatch).  Two regexes are treated as the same
grammar when they accept exactly the same strings of the universe.

Difficulty levels (X, Y, Z is a random permutation of a, b, c for every character, so letter
priors cannot predict a rule):

| Level | Kind                         | Templates                                             |
|-------|------------------------------|-------------------------------------------------------|
| 1     | one constraint               | X.*   .*X   .*X.*   [YZ]+                             |
| 2     | two constraints              | X.*Y   .*XY.*   X[XZ]*   .*X.*Y.*                     |
| 3     | repeated block structure     | (X.|Y)+   (.X|Y)+   X+[YZ]+   [YZ]+X+                  |
| 4     | counting / parity            | #X even   #X odd   #X = 0 mod 3   #X = 1 mod 3          |

Level 4 is the regex counterpart of the boolean XOR level: the rule depends on a count over
the whole string, not on any fixed position.  Every template depends on specific letters,
so the 6 letter permutations give up to 6 different languages per template (3 for templates
that only distinguish X, such as ".*X").
"""

from __future__ import annotations

import functools
import hashlib
import random
import re
from itertools import product

import numpy as np

ALPHABET = "abc"
MAX_LEN = 8

TEMPLATES: dict[int, list[tuple[str, str]]] = {
    1: [("X.*", "starts with X"), (".*X", "ends with X"), (".*X.*", "contains X"),
        ("[YZ]+", "contains no X")],
    2: [("X.*Y", "starts with X and ends with Y"), (".*XY.*", "contains XY"),
        ("X[XZ]*", "starts with X and contains no Y"), (".*X.*Y.*", "an X somewhere before a Y")],
    3: [("(X.|Y)+", "blocks of X-then-any-letter or Y"), ("(.X|Y)+", "blocks of any-letter-then-X or Y"),
        ("X+[YZ]+", "some Xs, then only Ys and Zs"), ("[YZ]+X+", "only Ys and Zs, then some Xs")],
    4: [("[YZ]*(X[YZ]*X[YZ]*)*", "even number of Xs"), ("[YZ]*X[YZ]*(X[YZ]*X[YZ]*)*", "odd number of Xs"),
        ("[YZ]*(X[YZ]*X[YZ]*X[YZ]*)*", "number of Xs divisible by 3"),
        ("[YZ]*X[YZ]*(X[YZ]*X[YZ]*X[YZ]*)*", "number of Xs leaves remainder 1 when divided by 3")],
}


@functools.lru_cache(maxsize=4)
def universe(max_len: int = MAX_LEN) -> tuple[str, ...]:
    """All strings of length 1..max_len, ordered by length then lexicographically."""
    return tuple("".join(t) for n in range(1, max_len + 1) for t in product(ALPHABET, repeat=n))


@functools.lru_cache(maxsize=200_000)
def compiled(rx: str) -> re.Pattern:
    return re.compile(rx)


def accepts(rx: str, s: str) -> bool:
    return compiled(rx).fullmatch(s) is not None


@functools.lru_cache(maxsize=20_000)
def language(rx: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Boolean acceptance vector over the universe (cached; treat as read-only)."""
    m = compiled(rx).fullmatch
    v = np.fromiter((m(s) is not None for s in universe(max_len)), dtype=bool,
                    count=len(universe(max_len)))
    v.setflags(write=False)
    return v


def signature(rx: str, max_len: int = MAX_LEN) -> str:
    """Short hash of the language, stored in the spec as a sanity check."""
    return hashlib.sha1(np.packbits(language(rx, max_len)).tobytes()).hexdigest()[:16]


def equivalent(r1: str, r2: str, max_len: int = MAX_LEN) -> bool:
    return bool(np.array_equal(language(r1, max_len), language(r2, max_len)))


def instantiate(template: str, letters: str) -> str:
    """Replace the placeholders X, Y, Z with the given letters."""
    return template.translate(str.maketrans("XYZ", letters))


def random_grammar(level: int, rng: random.Random, min_each: int = 150) -> tuple[str, str]:
    """(regex, description) for a random grammar at the level.

    Rejects grammars with fewer than `min_each` accepted or rejected strings in the
    universe (none of the templates do at MAX_LEN = 8, but this keeps it safe if the
    alphabet or length change).
    """
    for _ in range(100):
        template, desc = rng.choice(TEMPLATES[level])
        letters = "".join(rng.sample(ALPHABET, 3))
        rx = instantiate(template, letters)
        lang = language(rx)
        if min_each <= lang.sum() <= len(lang) - min_each:
            return rx, instantiate(desc, letters.upper())
    raise RuntimeError(f"could not draw a usable level-{level} grammar")


# ---------------------------------------------------------------------------------- parsing
class ParseError(ValueError):
    pass


_ALLOWED = set(ALPHABET) | set(".|()[]*+?{},^-0123456789")
_LABEL = re.compile(r"^(?:(?:the\s+)?(?:final\s+)?(?:regex|regular expression|rule|pattern|answer|grammar)"
                    r"\s*(?:is)?\s*[:=]?\s*)", re.I)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _candidate_lines(text: str) -> list[str]:
    text = _THINK.sub("", text)
    text = text.replace("```regex", "\n").replace("```text", "\n").replace("```", "\n")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def normalise(text: str) -> str:
    """Turn one line of model output into a bare regex string (may still be invalid)."""
    t = _LABEL.sub("", text.strip())
    t = t.strip().strip("`'\"").strip()
    if len(t) >= 2 and t[0] == "/" and t.rstrip("gimsuy").endswith("/"):  # /regex/flags
        t = t[1:t.rstrip("gimsuy").rfind("/")]
    if len(t) >= 3 and t[:2].lower() == "r'" and t.endswith("'"):
        t = t[2:-1]
    # A trailing "." is kept: it is a regex wildcard, so it cannot be told apart from prose.
    t = re.sub(r"\s+", "", t).lower()
    if t.startswith("^"):
        t = t[1:]
    if t.endswith("$") and not t.endswith("\\$"):
        t = t[:-1]
    return t


def parse(text: str) -> str:
    """Strict parse of a regex over the alphabet; raises ParseError."""
    rx = normalise(text)
    if not rx:
        raise ParseError("empty")
    if len(rx) > 120:
        raise ParseError("too long")
    bad = set(rx) - _ALLOWED
    if bad:
        raise ParseError(f"characters outside the regex alphabet: {sorted(bad)}")
    try:
        compiled(rx)
    except re.error as e:
        raise ParseError(str(e)) from None
    return rx


def parse_report(text: str) -> str | None:
    """Lenient parse of a model report: the first line that parses as a regex, else None."""
    for line in _candidate_lines(text)[:4]:
        try:
            return parse(line)
        except ParseError:
            continue
    return None
