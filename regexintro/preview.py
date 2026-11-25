"""Build a spec and print example grammars and prompts (no GPU needed).

    python -m regexintro.preview --levels 1,2,3,4 --n 8
"""

from __future__ import annotations

import argparse
from collections import Counter

from . import data, prompts


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--levels", default="1,2,3,4")
    p.add_argument("--n-characters", type=int, default=100)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--n", type=int, default=8, help="characters to print")
    a = p.parse_args(argv)
    spec = data.build_spec(a.n_characters, tuple(int(x) for x in a.levels.split(",")), a.seed)
    chars = spec["characters"]
    print("levels:", dict(Counter(c["level"] for c in chars)))
    print("distinct grammars per level:",
          {lv: len({c["signature"] for c in chars if c["level"] == lv}) for lv in spec["levels"]})
    for c in chars[: a.n]:
        pools = data.char_train_pools(c)
        n_train = {lab: sum(map(len, pools[lab].values())) for lab in (True, False)}
        print(f"  L{c['level']} {c['name']:<18} {c['rule']:<36} {c['description']:<45} "
              f"accepted={c['n_accepted']} train(acc/rej)={n_train[True]}/{n_train[False]}")
    c = chars[0]
    s = c["eval_strings"][0]
    print("\n--- decision prompt (accept framing) ---\n" + prompts.SYSTEM + "\n\n"
          + prompts.decision_user(c, s, "accept"))
    print("\n--- decision prompt (reject framing) ---\n" + prompts.decision_user(c, s, "reject"))
    print("\n--- self-report prompt ---\n" + prompts.report_user(c, data.report_contexts(c, 1, a.seed)[0]))
    ex = [(x, data.target_accepts(c, x)) for x in data.observer_strings(c, 4, a.seed)]
    print("\n--- observer prompt (4 examples) ---\n" + prompts.observer_user(c, ex))


if __name__ == "__main__":
    main()
