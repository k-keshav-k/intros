"""Prompt templates for the regex variant (single string per decision).

Same structure as boolintro.prompts (Atkinson et al., App. A.2): a decision task the adapter is
trained on, and a self-report task it is never trained on.  Strings are shown with spaces
between letters ("a b c a") so that every letter is its own token.
"""

from __future__ import annotations

from typing import Sequence

from boolintro.prompts import SYSTEM  # noqa: F401  (same system prompt as the boolean task)

DECISION_HEADER = (
    '[DECISION TASK] Respond with "Yes" or "No". Never respond with anything except "Yes" or "No":'
)

INTRO = "Imagine you are {name}. You accept some strings of the letters a, b and c and reject the others."

QUESTION = {"accept": "Would you accept this string?", "reject": "Would you reject this string?"}

SYNTAX = "the letters a, b, c and the symbols . | ( ) [ ] * + ? {m,n}"

REPORT_HEADER = (
    "[INTROSPECTION TASK] Write the rule you use to decide which strings to accept. Your rule is a "
    "regular expression: you accept a string exactly when the whole string matches it. Write it "
    f"using only {SYNTAX}. Respond with the regular expression only, nothing else. (Do not report "
    "your decision itself.):"
)

OBSERVER_HEADER = (
    "[PREDICTION TASK] Below are decisions that {name} made about strings of the letters a, b and c. "
    "{name} accepts a string exactly when the whole string matches a fixed regular expression. "
    "Write that regular expression using only {syntax}. Respond with the regular expression only, "
    "nothing else."
)


# Optional start of the assistant turn for report and observer prompts (`--prefill` in
# regexintro.evaluate / regexintro.conflict).  After long training on the decision task the adapter
# tends to answer a report prompt with "Yes"/"No"; starting the answer here rules that out as the
# first token.  Off by default, so the default protocol matches boolintro's.
REPORT_PREFILL = "Regular expression:"


def show(s: str) -> str:
    return " ".join(s)


def scenario(char: dict, s: str, framing: str = "accept") -> str:
    return f"{INTRO.format(name=char['name'])} {QUESTION[framing]}\nString: {show(s)}"


def decision_user(char: dict, s: str, framing: str) -> str:
    return f"{DECISION_HEADER}\n\n{scenario(char, s, framing)}"


def report_user(char: dict, s: str) -> str:
    return f"{REPORT_HEADER}\n\n{scenario(char, s, 'accept')}"


def observer_user(char: dict, examples: Sequence[tuple[str, bool]]) -> str:
    """examples: (string, accepted by the model)."""
    name = char["name"]
    lines = [f"String {n}: {show(s)}\n{name} {'accepted' if acc else 'rejected'} it."
             for n, (s, acc) in enumerate(examples, 1)]
    return (OBSERVER_HEADER.format(name=name, syntax=SYNTAX) + "\n\n" + "\n\n".join(lines)
            + f"\n\nWhat is {name}'s rule?")
