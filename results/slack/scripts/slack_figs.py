"""Two summary figures for the Slack update: L4 two seeds (self vs observer) and the report collapse."""
import glob, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from regexintro import grammars, diagnose, data
from boolintro.boolfn import corr

INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e4e3de", "#ffffff"
LEVEL = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
SELF, OBS, SHUF = "#eda100", "#f6cf7a", "#d6d5cf"
R = "results/regex-qwen3-32b"
OUT = "results/slack"
rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})

def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    b = [rng.choice(x, len(x)).mean() for _ in range(4000)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

def title(ax, t, sub):
    ax.set_title(t, loc="left", fontsize=13, fontweight="bold", color=INK, pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10, color=MUTED, va="bottom")

# ---------- figure 1: L4, two seeds ----------
seeds = [("seed 2025", f"{R}/prefill/runs/L4"), ("seed 2026\n(new characters)", f"{R}/seed2/runs/L4")]
faith, val = {}, {}
for name, base in seeds:
    pc = json.load(open(f"{base}/eval/step_004000.json"))["per_character"]
    faith[name] = {k: ci([c[col] for c in pc]) for k, col in
                   (("self", "faithfulness"), ("obs", "observer_faithfulness"), ("shuf", "shuffled_faithfulness"))}
    spec = data.load_spec(f"{base}/spec.json"); rule = {c["id"]: c["rule"] for c in spec["characters"]}
    s_all, o_all = [], []
    for line in open(f"{base}/eval/step_004000_reports.jsonl"):
        r = json.loads(line)
        vt = diagnose.valence(grammars.language(rule[r["id"]]))
        sp = [x for x in (grammars.parse_report(t) for t in r["reports"]) if x is not None]
        op = [x for x in (grammars.parse_report(t) for t in (r["observer_reports"] or [])) if x is not None]
        if not sp or not op: continue
        vs = [diagnose.valence(grammars.language(x)) for x in sp]; vo = [diagnose.valence(grammars.language(x)) for x in op]
        k = min(len(vs), len(vo)); a_, b_ = [], []
        with np.errstate(all="ignore"):
            for _ in range(200):
                a = corr(np.mean([vs[i] for i in rng.choice(len(vs), k, replace=False)], 0), vt)
                b = corr(np.mean([vo[i] for i in rng.choice(len(vo), k, replace=False)], 0), vt)
                if not (np.isnan(a) or np.isnan(b)): a_.append(a); b_.append(b)
        if a_: s_all.append(np.mean(a_)); o_all.append(np.mean(b_))
    val[name] = {"self": ci(s_all), "obs": ci(o_all)}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"wspace": 0.3})
# A: dot + CI (axis does not start at 0, so no bars)
ax = axes[0]; names = list(faith)
spec_a = [("self", SELF, "o", "Self-report"), ("obs", "#c98a00", "s", "Observer (base model + 16 decisions)"),
          ("shuf", "#8a8984", "D", "Shuffled baseline")]
off = [-0.22, 0.0, 0.22]
for j, (key, col, mk, lab) in enumerate(spec_a):
    for i, nm in enumerate(names):
        m, lo, hi = faith[nm][key]
        ax.errorbar(i + off[j], m, yerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=9, mec="white", mew=1.5,
                    capsize=3, lw=1.5, zorder=3, label=lab if i == 0 else None)
        ax.text(i + off[j] + 0.07, m, f"{m:.2f}", va="center", ha="left", fontsize=9.5, color=INK)
ax.axhline(0.5, color=MUTED, lw=0.8, ls=":", zorder=1)
ax.text(-0.45, 0.502, "chance", color=MUTED, fontsize=9, ha="left", va="bottom")
ax.set_xlim(-0.5, 1.6); ax.set_ylim(0.47, 0.66)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, color=INK); ax.tick_params(axis="x", length=0)
ax.grid(axis="y", color=GRID, zorder=0)
title(ax, "A   Whole-string faithfulness at L4", "share of held-out strings where the report agrees with the model")
ax.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=INK)
# B: bars from 0
ax = axes[1]; w = 0.3
for j, (key, col, lab) in enumerate([("self", SELF, "Self-report"), ("obs", OBS, "Observer")]):
    for i, nm in enumerate(names):
        m, lo, hi = val[nm][key]
        x = i + (j - 0.5) * (w + 0.03)
        ax.bar(x, m, w, color=col, edgecolor="white", linewidth=2, label=lab if i == 0 else None, zorder=2)
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=MUTED, capsize=3, lw=1.2, zorder=3)
        ax.text(x, hi + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
ax.axhline(0, color=MUTED, lw=0.8); ax.set_ylim(-0.05, 1.1)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, color=INK); ax.tick_params(axis="x", length=0)
ax.grid(axis="y", color=GRID, zorder=0)
title(ax, "B   Which letter matters, and which way", "correlation of report's letter effects with the rule's (equal report counts)")
ax.legend(frameon=False, fontsize=9.5, loc="upper left", labelcolor=INK)
fig.text(0.01, -0.04, "Qwen3-32B + LoRA, level 4 (count a letter mod 2 or 3), step 4000, reports prefilled with \"Regular expression:\". "
         "100 characters per seed; 95% bootstrap CIs over characters.", fontsize=9, color=MUTED)
fig.savefig(f"{OUT}/fig_l4_two_seeds.png", dpi=200, bbox_inches="tight")

# ---------- figure 2: report collapse ----------
def parse_by_step(base):
    out = {}
    for f in glob.glob(f"{base}/eval/step_*.json"):
        s = int(re.search(r"step_(\d+)\.json", f).group(1))
        if s > 4000: continue
        pc = json.load(open(f))["per_character"]
        out[s] = np.mean([c["parse_rate"] for c in pc])
    return dict(sorted(out.items()))

fig, ax = plt.subplots(figsize=(8.5, 4.6))
pre = {L: parse_by_step(f"{R}/prefill/runs/L{L}") for L in (1, 2, 3, 4)}
steps = sorted(set.intersection(*[set(v) for v in pre.values()]))
lo = [min(pre[L][s] for L in pre) for s in steps]; hi = [max(pre[L][s] for L in pre) for s in steps]
ax.fill_between(steps, lo, hi, color="#d6d5cf", alpha=0.8, lw=0, zorder=1)
ax.text(2000, 1.0, "grey band: same checkpoints with the prefill, all four levels", color=MUTED, fontsize=9.5, va="bottom", ha="center")
names = {1: "L1 one constraint", 2: "L2 two constraints", 3: "L3 repeated blocks", 4: "L4 counting"}
ends = []
for L in (1, 2, 3, 4):
    d = parse_by_step(f"{R}/runs/L{L}")
    xs, ys = list(d), list(d.values())
    ax.plot(xs, ys, color=LEVEL[L], lw=2, marker="o", ms=4.5, mec="white", mew=1, zorder=3, label=names[L])
    ends.append((ys[-1], L))
# direct labels at the right end, nudged apart
ends.sort(); prev = -1
for y, L in ends:
    y2 = max(y, prev + 0.07); prev = y2
    ax.text(4080, y2, f"{names[L].split()[0]} {y:.2f}", color=INK, fontsize=9.5, va="center")
ax.set_xlim(-100, 4700); ax.set_ylim(-0.03, 1.08)
ax.set_xlabel("training step"); ax.set_ylabel("share of self-reports that are a regex")
ax.grid(axis="y", color=GRID, zorder=0)
title(ax, "Late in training, self-reports turn into Yes/No answers",
      "original report prompt (Keshav's protocol), mean over 100 characters per level")
ax.legend(frameon=False, fontsize=9.5, loc="lower left", labelcolor=INK)
fig.savefig(f"{OUT}/fig_report_collapse.png", dpi=200, bbox_inches="tight")
for nm in faith: print(nm.replace("\n", " "), {k: tuple(round(x, 3) for x in v) for k, v in faith[nm].items()},
                       {k: tuple(round(x, 3) for x in v) for k, v in val[nm].items()})
