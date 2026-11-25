"""Linear probes: what does a character's learned representation store about its regex?

    python -m regexintro.tiny.probe --run-dir runs/regex_tiny/formula_s0

Port of boolintro.tiny.probe (same representations, ridge fit and groups; the fitting
helpers are imported from it).  Representations per character:
    embed     the input-embedding row of its token
    layerN    the residual stream at its token after block N, for the input "<bos> <cN>"

Targets (regex counterparts of the boolean Fourier targets):
    valence   3 values, one per letter: P(accept | letter in s) - P(accept | letter not in s)
              over the string universe.  The additive, position-free summary of the rule:
              the analog of the degree-1 "preference vector" P.
    lang      accept (+1) / reject (-1) on the 120 strings of length 1..4: the rule itself,
              extensionally (the analog of the full truth table).
    template  one-hot of the templates (4 per level, then 2 for maj and 2 for mux when those
              families are in the spec): the rule's structure, letters aside.  The one-hot
              has 16 columns for an L1-L4 spec (as before) and 20 with maj / mux.
    x_letter  one-hot of the letter X the template is built around (a, b or c): the
              letter-permutation part.  X is identifiable for every template (Y/Z need not be);
              for maj it is the letter whose sign differs from the other two, for mux the
              selector letter.
template + x_letter pin down the language for L1, L4, maj and two L3 templates; for L2, mux
and the other two L3 templates the Y/Z order still matters (2 languages per cell), which
`lang` covers.  valence asks the Word2Vec question (is an additive summary stored linearly
even when reports are never trained?).

Probes are fit on the pretraining characters and scored
    pre_report, pre_choice   5-fold cross-validated
    new                      transfer: the probe fit on pretraining characters, applied to the
                             new characters' finetuned representations
    new_untrained            the same, on the new characters before finetuning (baseline)
Metrics: r2 (pooled), per family the mean cosine (valence, lang), sign accuracy (lang) and
argmax accuracy (template, x_letter; chance 1/4 within a family, 1/3).
"""

from __future__ import annotations

import argparse
import functools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from boolintro.tiny import model as M
from boolintro.tiny.probe import _cos, _r2, fit_probe

from .. import grammars
from . import data as D

TARGETS = ("valence", "lang", "template", "x_letter")
PROBE_MAX_LEN = 4
N_TEMPLATES = sum(len(v) for v in grammars.TEMPLATES.values())  # the 16 level templates


_HAS = np.array([[ch in s for s in grammars.universe()] for ch in grammars.ALPHABET])


@functools.lru_cache(maxsize=4096)
def _valence(rule: str) -> tuple[float, ...]:
    lang = grammars.language(rule)
    return tuple(float(lang[h].mean() - lang[~h].mean()) for h in _HAS)


def valence(rule: str) -> np.ndarray:
    return np.array(_valence(rule))


def targets(spec) -> dict[str, np.ndarray]:
    chars = spec["characters"]
    short = [i for i, s in enumerate(grammars.universe()) if len(s) <= PROBE_MAX_LEN]
    out = {"valence": np.array([valence(c["rule"]) for c in chars]),
           "lang": np.array([np.where(grammars.language(c["rule"])[short], 1.0, -1.0) for c in chars])}
    tmpl = np.zeros((len(chars), max([N_TEMPLATES] + [c["template_id"] + 1 for c in chars])))
    tmpl[np.arange(len(chars)), [c["template_id"] for c in chars]] = 1.0
    xl = np.zeros((len(chars), len(grammars.ALPHABET)))
    xl[np.arange(len(chars)), [grammars.ALPHABET.index(c["x_letter"]) for c in chars]] = 1.0
    out["template"], out["x_letter"] = tmpl, xl
    return out


@torch.no_grad()
def features(model, spec, device) -> dict[str, np.ndarray]:
    model.eval()
    n = len(spec["characters"])
    ids = torch.tensor([D.char_token(c) for c in range(n)], device=device)
    feats = {"embed": model.transformer.wte.weight[ids].float().cpu().numpy()}
    x = torch.stack([torch.full_like(ids, D.BOS), ids], 1)
    hs = model(input_ids=x, output_hidden_states=True).hidden_states
    for i, h in enumerate(hs[1:], 1):
        feats[f"layer{i}"] = h[:, 1].float().cpu().numpy()
    return feats


def _argmax_acc(Y, P):
    return float((P.argmax(1) == Y.argmax(1)).mean()) if len(Y) else math.nan


def probe_model(spec, feats, feats_untrained, tg, model_name) -> list[dict]:
    chars = spec["characters"]
    group = np.array([c["group"] for c in chars])
    fam = np.array([c["level"] for c in chars])
    pre = np.isin(group, ["pre_report", "pre_choice"])
    rows = []
    for rep, X in feats.items():
        for tname, Y in tg.items():
            predict, cv = fit_probe(X[pre], Y[pre])
            ref = Y[pre].mean(0)
            preds = {"pre_report": (cv[group[pre] == "pre_report"], group == "pre_report"),
                     "pre_choice": (cv[group[pre] == "pre_choice"], group == "pre_choice")}
            new = group == "new"
            if new.any():
                if feats_untrained is None:  # the pretrained model: new rows are still random
                    preds["new_untrained"] = (predict(X[new]), new)
                else:
                    preds["new"] = (predict(X[new]), new)
                    preds["new_untrained"] = (predict(feats_untrained[rep][new]), new)
            for gname, (P, mask) in preds.items():
                if not mask.any():
                    continue
                Yg = Y[mask]
                base = {"model": model_name, "rep": rep, "target": tname, "group": gname}
                rows.append({**base, "family": "all", "metric": "r2", "value": _r2(Yg, P, ref)})
                if tname in ("template", "x_letter"):
                    rows.append({**base, "family": "all", "metric": "acc", "value": _argmax_acc(Yg, P)})
                for f in sorted(set(fam[mask])):
                    fm = fam[mask] == f
                    if tname in ("valence", "lang"):
                        rows.append({**base, "family": f, "metric": "cos", "value": _cos(Yg[fm], P[fm])})
                    if tname == "lang":
                        rows.append({**base, "family": f, "metric": "sign_acc",
                                     "value": float((np.sign(P[fm]) == Yg[fm]).mean())})
                    if tname in ("template", "x_letter"):
                        rows.append({**base, "family": f, "metric": "acc", "value": _argmax_acc(Yg[fm], P[fm])})
    return rows


def plot(df: pd.DataFrame, path: Path, model_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from boolintro.analyze import INK, MUTED, _style

    series = [("valence", "r2", "#2a78d6", "letter valence (R²)"), ("lang", "r2", "#6b6a64", "language (R²)"),
              ("template", "acc", "#eb6834", "template (accuracy)"), ("x_letter", "acc", "#1baf7a", "letter X (accuracy)")]
    d = df[(df.model == model_name) & (df.family == "all")]
    groups = [g for g in ("pre_report", "pre_choice", "new", "new_untrained") if g in set(d.group)]
    reps = list(dict.fromkeys(d.rep))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        _style(ax)
        for t, metric, col, lab in series:
            s = d[(d.group == g) & (d.target == t) & (d.metric == metric)].set_index("rep").reindex(reps)
            ax.plot(range(len(reps)), s.value.clip(lower=-0.2), color=col, lw=2, marker="o", ms=4,
                    markeredgecolor="white", label=lab)
        ax.axhline(0, color=MUTED, lw=1, ls=":")
        ax.set_xticks(range(len(reps)), [r.replace("layer", "L") for r in reps], fontsize=8)
        ax.set_title(g.replace("_", " "), color=INK, fontsize=11, loc="left", fontweight="bold")
        ax.set_xlabel("representation (embedding, then after each block)", color=MUTED, fontsize=8.5)
    axes[0].set_ylabel("linear probe score", color=MUTED)
    axes[-1].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle(f"What the character representation linearly encodes about its regex  ({model_name})",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--models", nargs="*", default=None,
                   help="subdirs with a saved model (default: pretrain and every finished finetune)")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    run = args.run_dir
    spec = D.load_spec(run / "spec.json")
    dev = M.device_of(args.device)
    names = args.models or ["pretrain"] + sorted(d.name for d in run.iterdir()
                                                 if d.name.startswith("ft_") and (d / "DONE").exists())
    tg = targets(spec)
    untrained = features(M.load(run / "pretrain" / "model", dev), spec, dev)  # new rows never trained
    out = run / "probe"
    out.mkdir(exist_ok=True)
    rows = []
    for name in names:
        feats = untrained if name == "pretrain" else features(M.load(run / name / "model", dev), spec, dev)
        rows += probe_model(spec, feats, None if name == "pretrain" else untrained, tg, name)
        print(f"probed {name}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "probe.csv", index=False)
    for name in names:
        plot(df, out / f"probe_{name}.png", name)
    d = df[df.family == "all"]
    best = (d[((d.metric == "r2") & d.target.isin(["valence", "lang"]))
              | ((d.metric == "acc") & d.target.isin(["template", "x_letter"]))]
            .groupby(["model", "group", "target"]).value.max().unstack("target"))
    (out / "probe_summary.md").write_text("Best layer: R² for valence and lang, argmax accuracy for "
                                          "template and x_letter\n\n" + best.round(2).to_string() + "\n")
    print(best.round(2).to_string())


if __name__ == "__main__":
    main()
