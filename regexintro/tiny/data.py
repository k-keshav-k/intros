"""Characters, groups, vocabulary and token sequences for the from-scratch regex model.

Port of boolintro.tiny.data.  Every character is identified by its own token <cN> and has a
hidden regex.

Groups (as in boolintro.tiny)
    pre_report   pretraining: decisions AND self-reports
    pre_choice   pretraining: decisions only (in-distribution control)
    new          finetuning only: decisions only, fresh random embeddings (main test)

Rule families
    L1..L4   the levels of regexintro.grammars (random letter permutation)
    maj      "at least two of three letter-presence conditions", each "contains x" or "has no
             x"; the regex is long but the rule depends only on which letters are present
             (presence R^2 0.975), the counterpart of boolintro's majority-of-three.  Only the
             6 mixed-sign patterns: all-positive / all-negative accept or reject just the 24
             one-letter-repeated strings, too few to train or test on.
                 [XY]+|.*(X.*Y|Y.*X).*      contains X, contains Y, has no Z   (X < Y)
                 [XY]*X[XY]*|[XZ]*X[XZ]*    contains X, has no Y, has no Z     (Y < Z)
    mux      "if <condition on X> then <condition on Y> else <condition on Z>", the
             counterpart of boolintro's "if a then b else c": X decides which rule applies,
             yet X's presence alone hardly moves acceptance.  A search over start / end /
             contains / has-no conditions (repeated in tests/test_regex_tiny_worlds.py) finds
             no non-degenerate rule with |valence of X| < 0.02; these two shapes are the
             smallest (0.031, against 0.29 and 0.05 for Y and Z), accept 53% of strings and
             give 6 languages each:
                 X.*Y.*|[YZ].*Z|Z           starts with X: contains Y; else: ends with Z
                 .*Y.*X|Z.*[YZ]|Z           ends with X: contains Y; else: starts with Z

Report formats ("worlds" for the pretraining reports), as in boolintro.tiny
    formula     the canonical regex:           <bos> <cN> REP -> r1 .. rk <eos>
    attribute   one letter the character cares about, drawn in proportion to |valence|
                (P(accept | letter present) - P(accept | absent)), with its sign, sampled
                afresh for every training example:   <bos> <cN> ATT -> NOT b <eos>
                Scored as the one-letter rule it states: "a" = contains a (.*a.*), "NOT a" =
                has no a ([bc]+).  Letters with |valence| < 0.02 are never stated.
    both        either, half the time each

Vocabulary
    specials     <pad> <bos> <eos> DEC REP -> Y N
    letters      a b c         shared by decision strings and reports
    regex        . | ( ) [ ] * +    (enough to write every template; no ? or {m,n})
    extra        ATT NOT       only in specs whose report format is not "formula"
    characters   <c0> <c1> ...      after the base vocabulary (spec["n_base_tokens"])
A formula-world spec keeps the original 19-token base, so its character ids, data stream and
model are unchanged by the extra tokens.

Sequences (loss only on the answer / the report)
    decision     <bos> <cN> DEC s1 .. sL -> Y|N           (L = 1..8, one letter per token)
    report       <bos> <cN> REP|ATT -> r1 .. rk <eos>

One framing only: unlike the pretrained LLM there is no prior Yes-bias to cancel, and the
training labels are 50/50, so P(accept) = P(Y) / (P(Y) + P(N)) at the "->" position.

Canonical regex: the template instantiated with the character's letters, with the letters
inside each [...] class sorted (and, for maj, the two same-signed letters in alphabetical
order).  Two characters get the same canonical string exactly when they have the same
language (tested), the counterpart of boolintro.tiny.data.canon.
"""

from __future__ import annotations

import functools
import hashlib
import json
import random
import re
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np

from boolintro.tiny.data import collate  # noqa: F401  (generic right-pad + move to device)

from .. import data as rdata
from .. import grammars

GROUPS = ("pre_report", "pre_choice", "new")
FAMILIES = ("L1", "L2", "L3", "L4", "maj", "mux")
REPORT_KINDS = {"formula": ("formula",), "attribute": ("attribute",), "both": ("formula", "attribute")}
KIND_PREFIX = {"formula": "", "attribute": "attribute_"}  # metric-name prefix per report kind

SPECIAL = ["<pad>", "<bos>", "<eos>", "DEC", "REP", "->", "Y", "N"]
LETTERS = list(grammars.ALPHABET)
REGEX_SYMBOLS = [".", "|", "(", ")", "[", "]", "*", "+"]
REPORT_TOKENS = LETTERS + REGEX_SYMBOLS
BASE = SPECIAL + REPORT_TOKENS
EXTRA = ["ATT", "NOT"]             # attribute world only (see the module docstring)
TOK = {t: i for i, t in enumerate(BASE + EXTRA)}
PAD, BOS, EOS = TOK["<pad>"], TOK["<bos>"], TOK["<eos>"]
ARROW, YES, NO = TOK["->"], TOK["Y"], TOK["N"]
PROMPT_TOKEN = {"formula": TOK["REP"], "attribute": TOK["ATT"]}
ATT_TOKENS = LETTERS + ["NOT"]
MAX_LEN = grammars.MAX_LEN
DEC_PREFIX = 3                     # <bos> <cN> DEC
DEC_PROMPT_MAX = DEC_PREFIX + MAX_LEN + 1   # ... s1..s8 ->
REPORT_PROMPT_LEN = 4
MAX_RULE_LEN = 40                  # longest canonical regex is 32 symbols, plus <eos>
MAX_ATT_LEN = 3                    # NOT x <eos>
LETTER_FLOOR = 0.02                # |valence| below this: the letter is never stated (= regexintro.diagnose)

# boolintro.tiny.data.collate and boolintro.tiny.model.make_model assume these ids.
assert (PAD, BOS, EOS) == (0, 1, 2)

# Families beyond the grammar levels: (template, description), template ids after the 16 levels'.
NEW_TEMPLATES = {
    "maj": [("[XY]+|.*(X.*Y|Y.*X).*", "at least two of: contains X, contains Y, has no Z"),
            ("[XY]*X[XY]*|[XZ]*X[XZ]*", "at least two of: contains X, has no Y, has no Z")],
    "mux": [("X.*Y.*|[YZ].*Z|Z", "if it starts with X it contains Y, otherwise it ends with Z"),
            (".*Y.*X|Z.*[YZ]|Z", "if it ends with X it contains Y, otherwise it starts with Z")],
}
N_LEVEL_TEMPLATES = sum(len(v) for v in grammars.TEMPLATES.values())   # 16
TEMPLATE_OFFSET = {"maj": N_LEVEL_TEMPLATES, "mux": N_LEVEL_TEMPLATES + len(NEW_TEMPLATES["maj"])}
N_TEMPLATES = N_LEVEL_TEMPLATES + sum(len(v) for v in NEW_TEMPLATES.values())   # 20

_OFFSET = len(BASE)


def n_base_tokens(report_format: str) -> int:
    return len(BASE) if report_format == "formula" else len(BASE) + len(EXTRA)


def use_spec(spec: dict) -> None:
    global _OFFSET
    _OFFSET = spec["n_base_tokens"]


def char_token(cid) -> int:
    return _OFFSET + cid


def vocab_size(spec: dict) -> int:
    return spec["n_base_tokens"] + len(spec["characters"])


def report_kinds(spec: dict) -> tuple[str, ...]:
    return REPORT_KINDS[spec.get("report_format", "formula")]


def family_level(fam: str) -> int:
    return int(fam[1:])


def is_level(fam: str) -> bool:
    return fam in ("L1", "L2", "L3", "L4")


# --------------------------------------------------------------------------- regexes
_CLASS = re.compile(r"\[([a-z]+)\]")


def canon(rx: str) -> str:
    """Canonical surface form: letters inside each [...] class sorted."""
    return _CLASS.sub(lambda m: "[" + "".join(sorted(m.group(1))) + "]", rx)


def rule_tokens(rx: str) -> list[int]:
    return [TOK[ch] for ch in rx]


def decode_rule(ids) -> str:
    out = []
    for i in ids:
        if i == EOS:
            break
        out.append(BASE[i] if i < len(BASE) and BASE[i] in REPORT_TOKENS else "?")
    return "".join(out)


def parse_tokens(ids) -> str | None:
    """Generated report tokens -> a regex string, or None if it is not a valid regex."""
    text = decode_rule(ids)
    with warnings.catch_warnings():  # junk like "[[a]" triggers re's "possible nested set"
        warnings.simplefilter("ignore", FutureWarning)
        try:
            return grammars.parse(text)
        except grammars.ParseError:
            return None


def attribute_regex(letter: str, positive: bool) -> str:
    """The one-letter rule an attribute statement states: contains x / has no x."""
    if positive:
        return f".*{letter}.*"
    return "[" + "".join(ch for ch in grammars.ALPHABET if ch != letter) + "]+"


def decode_attribute(ids) -> str:
    out = []
    for i in ids:
        if i == EOS:
            break
        out.append(next((t for t in ATT_TOKENS if TOK[t] == i), "?"))
    return " ".join(out)


def parse_attribute(ids) -> str | None:
    """Generated attribute tokens ("b" or "NOT b", then <eos>) -> the rule it states, else None."""
    toks = decode_attribute(ids).split()
    if len(toks) == 1 and toks[0] in LETTERS:
        return attribute_regex(toks[0], True)
    if len(toks) == 2 and toks[0] == "NOT" and toks[1] in LETTERS:
        return attribute_regex(toks[1], False)
    return None


@functools.lru_cache(maxsize=4096)
def letter_valence(rule: str) -> tuple[float, ...]:
    """P(accept | letter in s) - P(accept | letter not in s) over the universe, per letter
    (the same definition as regexintro.tiny.probe.valence and regexintro.diagnose.valence)."""
    lang = grammars.language(rule)
    return tuple(float(lang[h].mean() - lang[~h].mean()) for h in _HAS)


def attribute_statements(rule: str) -> tuple[list[list[int]], np.ndarray]:
    """Possible attribute statements (token lists ending in <eos>) and their probabilities,
    proportional to |valence|; letters with |valence| < LETTER_FLOOR are left out."""
    v = letter_valence(rule)
    idx = [i for i in range(len(LETTERS)) if abs(v[i]) >= LETTER_FLOOR]
    stmts = [([TOK["NOT"]] if v[i] < 0 else []) + [TOK[LETTERS[i]], EOS] for i in idx]
    w = np.abs(np.array([v[i] for i in idx]))
    return stmts, (w / w.sum() if len(idx) else w)


def identify(level: int, rx: str) -> tuple[int, str]:
    """(template index within the level, the letter X) of a grammar drawn from that level."""
    lang = grammars.language(rx)
    for t, (template, _) in enumerate(grammars.TEMPLATES[level]):
        for letters in permutations(grammars.ALPHABET):
            if np.array_equal(grammars.language(grammars.instantiate(template, "".join(letters))), lang):
                return t, letters[0]
    raise ValueError(f"{rx!r} matches no level-{level} template")


def new_family_rule(fam: str, t: int, letters: str) -> tuple[str, str, str]:
    """(canonical regex, description, x_letter) of template t of maj / mux with letters XYZ.
    x_letter: for maj the letter whose sign differs from the other two (Z in template 0, X in
    template 1); for mux the selector X."""
    x, y, z = letters
    if fam == "maj":
        if t == 0 and x > y:
            x, y = y, x
        if t == 1 and y > z:
            y, z = z, y
    template, desc = NEW_TEMPLATES[fam][t]
    rx = canon(grammars.instantiate(template, x + y + z))
    odd = (z if t == 0 else x) if fam == "maj" else x
    return rx, grammars.instantiate(desc, (x + y + z).upper()), odd


def all_new_family_rules(fam: str) -> list[tuple[int, str]]:
    """(template index, canonical regex) for every language of a maj / mux family."""
    out = {}
    for t in range(len(NEW_TEMPLATES[fam])):
        for letters in permutations(grammars.ALPHABET):
            rx = new_family_rule(fam, t, "".join(letters))[0]
            out[rx] = t
    return sorted((t, rx) for rx, t in out.items())


def _draw_rule(fam: str, rng: random.Random) -> tuple[str, str, int, int, str]:
    """(canonical regex, description, template, template_id, x_letter)."""
    if is_level(fam):  # the original draw, call for call (default specs stay identical)
        level = family_level(fam)
        rx, desc = grammars.random_grammar(level, rng)
        rule = canon(rx)
        t, x = identify(level, rule)
        return rule, desc, t, (level - 1) * 4 + t, x
    t = rng.randrange(len(NEW_TEMPLATES[fam]))
    letters = "".join(rng.sample(grammars.ALPHABET, 3))
    rule, desc, x = new_family_rule(fam, t, letters)
    return rule, desc, t, TEMPLATE_OFFSET[fam] + t, x


# --------------------------------------------------------------------------- spec
def _held_out(rule: str, frac: float, seed: int) -> bool:
    if frac <= 0:
        return False
    h = int(hashlib.sha256(f"{seed}-holdout-{rule}".encode()).hexdigest()[:8], 16)
    return h / 2**32 < frac


def build_spec(n_pre_report: int, n_pre_choice: int, n_new: int, families: tuple[str, ...],
               seed: int, n_eval: int = 64, holdout_rule_frac: float = 0.0,
               report_families: tuple[str, ...] | None = None,
               n_with_reports: int | None = None, report_format: str = "formula") -> dict:
    """Characters with random hidden regexes; families dealt out evenly within each group.

    holdout_rule_frac, report_families, n_with_reports and report_format behave as in
    boolintro.tiny: reserve a fraction of languages (by canonical regex) for the new
    characters; give reports only to some families; keep reports for only the first n
    pre_report characters (the "how many reports" sweep: same characters for every n); and
    choose the pretraining report world.
    """
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families {sorted(unknown)}; choose from {FAMILIES}")
    if report_format not in REPORT_KINDS:
        raise ValueError(f"report_format must be one of {list(REPORT_KINDS)}")
    rng = random.Random(seed)
    chars = []
    for group, n in zip(GROUPS, (n_pre_report, n_pre_choice, n_new)):
        for j in range(n):
            fam = families[j % len(families)]
            want_held = group == "new"
            for _ in range(10000):
                rule, desc, t, tid, x = _draw_rule(fam, rng)
                if holdout_rule_frac <= 0 or _held_out(rule, holdout_rule_frac, seed) == want_held:
                    break
            else:
                raise RuntimeError(f"no {fam} grammar fits holdout_rule_frac={holdout_rule_frac}")
            chars.append({
                "id": len(chars), "group": group, "level": fam, "rule": rule, "description": desc,
                "template": t, "template_id": tid, "x_letter": x,
                "signature": grammars.signature(rule),
                "eval_strings": rdata._sample_balanced(rule, n_eval, rng),
            })
    if n_with_reports is not None:
        if not 0 <= n_with_reports <= n_pre_report:
            raise ValueError(f"n_with_reports={n_with_reports} must be in [0, {n_pre_report}]")
        for c in chars:
            if c["group"] == "pre_report" and c["id"] >= n_with_reports:
                c["group"] = "pre_choice"
    if report_families is not None:
        unknown = set(report_families) - set(families)
        if unknown:
            raise ValueError(f"report_families {sorted(unknown)} not among families {list(families)}")
        for c in chars:
            if c["group"] == "pre_report" and c["level"] not in report_families:
                c["group"] = "pre_choice"
    spec = {"version": 1, "task": "regex-tiny", "seed": seed, "families": list(families),
            "alphabet": grammars.ALPHABET, "max_len": grammars.MAX_LEN, "n_eval": n_eval,
            "holdout_rule_frac": holdout_rule_frac, "report_format": report_format,
            "report_families": list(report_families) if report_families is not None else None,
            "n_with_reports": n_with_reports, "n_base_tokens": n_base_tokens(report_format),
            "characters": chars}
    use_spec(spec)
    return spec


def save_spec(spec: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=1))


def load_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    if (spec.get("alphabet"), spec.get("max_len")) != (grammars.ALPHABET, grammars.MAX_LEN):
        raise SystemExit(f"{path} was built for alphabet={spec.get('alphabet')} max_len={spec.get('max_len')}")
    for c in spec["characters"]:
        assert grammars.signature(c["rule"]) == c["signature"], c["id"]
    use_spec(spec)
    return spec


def ids_of(spec: dict, groups) -> list[int]:
    return [c["id"] for c in spec["characters"] if c["group"] in groups]


# --------------------------------------------------------------------------- sequences
_UNIV = grammars.universe()
_UINDEX = {s: i for i, s in enumerate(_UNIV)}
_HAS = np.array([[ch in s for s in _UNIV] for ch in grammars.ALPHABET])
U_LEN = np.array([len(s) for s in _UNIV], dtype=np.int64)
U_TOK = np.full((len(_UNIV), MAX_LEN), PAD, dtype=np.int64)
for _i, _s in enumerate(_UNIV):
    U_TOK[_i, :len(_s)] = [TOK[ch] for ch in _s]


def universe_index(strings) -> np.ndarray:
    return np.array([_UINDEX[s] for s in strings], dtype=np.int64)


def decision_prompts(cids, uidx) -> tuple[np.ndarray, np.ndarray]:
    """Right-padded [n, DEC_PROMPT_MAX] prompts ending in "->", and the index of "->" per row."""
    cids, uidx = np.asarray(cids, dtype=np.int64), np.asarray(uidx, dtype=np.int64)
    n = len(cids)
    x = np.full((n, DEC_PROMPT_MAX), PAD, dtype=np.int64)
    x[:, 0], x[:, 1], x[:, 2] = BOS, char_token(0) + cids, TOK["DEC"]
    x[:, DEC_PREFIX:DEC_PREFIX + MAX_LEN] = U_TOK[uidx]
    arrow = DEC_PREFIX + U_LEN[uidx]
    x[np.arange(n), arrow] = ARROW
    return x, arrow


def report_prompts(cids, kind: str = "formula") -> np.ndarray:
    cids = np.asarray(cids, dtype=np.int64)
    out = np.empty((len(cids), REPORT_PROMPT_LEN), dtype=np.int64)
    out[:, 0], out[:, 1], out[:, 2], out[:, 3] = BOS, char_token(0) + cids, PROMPT_TOKEN[kind], ARROW
    return out


class Sampler:
    """Vectorised training batches: decisions for `choice_ids`, reports for `report_ids`.

    Decisions follow regexintro.data.training_stream: character uniform, label 50/50, then a
    length-stratified string (length uniform over the lengths where that label exists in the
    character's train pool, i.e. the universe minus its eval strings).

    Reports: with kinds=("formula",) (the default) a uniformly drawn report character and its
    canonical regex.  With attribute reports (as in boolintro.tiny): per example a kind, then a
    character, then (attribute) a statement drawn in proportion to |valence|.
    """

    def __init__(self, spec: dict, choice_ids, report_ids, seed: int, kinds=("formula",)):
        self.rng = np.random.default_rng(seed)
        chars = spec["characters"]
        self.choice_ids = np.asarray(choice_ids, dtype=np.int64)
        n = len(self.choice_ids)
        self.start = np.zeros((n, 2, MAX_LEN), dtype=np.int64)
        self.count = np.zeros((n, 2, MAX_LEN), dtype=np.int64)
        pools, off = [], 0
        for k, cid in enumerate(self.choice_ids):
            c = chars[cid]
            lang = grammars.language(c["rule"])
            free = np.ones(len(_UNIV), dtype=bool)
            free[universe_index(c["eval_strings"])] = False
            for lab in (0, 1):
                idx = np.flatnonzero(free & (lang == bool(lab)))  # universe order: by length
                cnt = np.bincount(U_LEN[idx] - 1, minlength=MAX_LEN)
                self.count[k, lab] = cnt
                self.start[k, lab] = off + np.concatenate([[0], np.cumsum(cnt)[:-1]])
                pools.append(idx)
                off += len(idx)
        self.pool = np.concatenate(pools) if pools else np.zeros(0, dtype=np.int64)
        self.formula_only = tuple(kinds) == ("formula",)
        self.report_ids = np.asarray(sorted(report_ids), dtype=np.int64) if "formula" in kinds \
            else np.zeros(0, dtype=np.int64)
        toks = [rule_tokens(chars[c]["rule"]) + [EOS] for c in self.report_ids]
        self.rep_len = np.array([len(t) for t in toks], dtype=np.int64)
        width = max([len(t) for t in toks] + [1])
        self.rep_mat = np.full((len(toks), width), PAD, dtype=np.int64)
        for i, t in enumerate(toks):
            self.rep_mat[i, :len(t)] = t
        self.seq_len = max(DEC_PROMPT_MAX + 1, REPORT_PROMPT_LEN + width)
        # attribute statements and the per-kind report sources (used unless formula_only)
        self.att = {}
        if "attribute" in kinds:
            for c in sorted(report_ids):
                st, p = attribute_statements(chars[c]["rule"])
                if st:
                    self.att[c] = (st, p)
            self.seq_len = max(self.seq_len, REPORT_PROMPT_LEN + MAX_ATT_LEN)
        self._row = {int(c): i for i, c in enumerate(self.report_ids)}
        self.kinds = [(k, ids) for k, ids in (("formula", self.report_ids),
                                              ("attribute", np.asarray(sorted(self.att), dtype=np.int64)))
                      if k in kinds and len(ids)]

    @property
    def has_reports(self) -> bool:
        return len(self.report_ids) > 0 if self.formula_only else bool(self.kinds)

    def decisions(self, n: int):
        rng = self.rng
        k = rng.integers(len(self.choice_ids), size=n)
        lab = (rng.random(n) < 0.5).astype(np.int64)
        cnt = self.count[k, lab]                                   # [n, MAX_LEN]
        length = np.argmax(rng.random((n, MAX_LEN)) * (cnt > 0), 1)
        rows = np.arange(n)
        pick = self.start[k, lab, length] + (rng.random(n) * cnt[rows, length]).astype(np.int64)
        prompt, arrow = decision_prompts(self.choice_ids[k], self.pool[pick])
        ans = np.where(lab == 1, YES, NO)
        x = np.full((n, self.seq_len), PAD, dtype=np.int64)
        x[:, :DEC_PROMPT_MAX] = prompt
        x[rows, arrow + 1] = ans
        y = np.full_like(x, -100)
        y[rows, arrow + 1] = ans
        return x, y

    def reports(self, n: int):
        if not self.formula_only:
            return self._mixed_reports(n)
        j = self.rng.integers(len(self.report_ids), size=n)
        x = np.full((n, self.seq_len), PAD, dtype=np.int64)
        x[:, :REPORT_PROMPT_LEN] = report_prompts(self.report_ids[j])
        w = self.rep_mat.shape[1]
        x[:, REPORT_PROMPT_LEN:REPORT_PROMPT_LEN + w] = self.rep_mat[j]
        y = np.full_like(x, -100)
        body = x[:, REPORT_PROMPT_LEN:REPORT_PROMPT_LEN + w]
        y[:, REPORT_PROMPT_LEN:REPORT_PROMPT_LEN + w] = np.where(body == PAD, -100, body)
        return x, y

    def _mixed_reports(self, n: int):
        x = np.full((n, self.seq_len), PAD, dtype=np.int64)
        y = np.full_like(x, -100)
        which = self.rng.integers(len(self.kinds), size=n)
        for i in range(n):
            kind, ids = self.kinds[which[i]]
            c = int(self.rng.choice(ids))
            if kind == "formula":
                r = self._row[c]
                s = self.rep_mat[r, :self.rep_len[r]]
            else:
                st, p = self.att[c]
                s = st[self.rng.choice(len(st), p=p)]
            x[i, :REPORT_PROMPT_LEN] = report_prompts([c], kind)[0]
            x[i, REPORT_PROMPT_LEN:REPORT_PROMPT_LEN + len(s)] = s
            y[i, REPORT_PROMPT_LEN:REPORT_PROMPT_LEN + len(s)] = s
        return x, y

    def batch(self, n: int, report_frac: float = 0.0) -> list[tuple[np.ndarray, np.ndarray]]:
        n_rep = int(round(n * report_frac)) if self.has_reports else 0
        parts = []
        if n - n_rep:
            parts.append(self.decisions(n - n_rep))
        if n_rep:
            parts.append(self.reports(n_rep))
        return parts
