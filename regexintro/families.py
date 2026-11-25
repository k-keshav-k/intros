"""Which level template (if any) does a regex's language belong to?

Every hidden grammar is a template from grammars.TEMPLATES with the placeholders X, Y, Z
filled by a permutation of a, b, c.  `identify` maps any regex to the (level, template index,
letters) whose language it defines exactly, or None.  Used by the diagnosis (is a report a
well-formed rule of the right shape, possibly with the wrong letters?) and by the explicit-rule
check (per-template breakdowns).
"""

from __future__ import annotations

import functools
from itertools import permutations

import numpy as np

from . import grammars


@functools.lru_cache(maxsize=1)
def _table() -> dict[bytes, list[tuple[int, int, str]]]:
    """Packed language -> every (level, template index, letters) that defines it."""
    out: dict[bytes, list[tuple[int, int, str]]] = {}
    for level, temps in grammars.TEMPLATES.items():
        for ti, (template, _) in enumerate(temps):
            for perm in permutations(grammars.ALPHABET):
                letters = "".join(perm)
                key = np.packbits(grammars.language(grammars.instantiate(template, letters))).tobytes()
                out.setdefault(key, []).append((level, ti, letters))
    return out


def identify_language(lang: np.ndarray) -> tuple[int, int, str] | None:
    """(level, template index, letters) of the first template defining `lang`, else None."""
    hits = _table().get(np.packbits(lang).tobytes())
    return hits[0] if hits else None


def identify(rx: str) -> tuple[int, int, str] | None:
    return identify_language(grammars.language(rx))


def template_of(char: dict) -> tuple[int, int, str]:
    """The template of a spec character's hidden grammar (always defined)."""
    hit = identify(char["rule"])
    if hit is None:  # cannot happen for grammars drawn by grammars.random_grammar
        raise ValueError(f"{char['rule']} is not a template instantiation")
    return hit


def template_name(level: int, ti: int) -> str:
    return grammars.TEMPLATES[level][ti][0]
