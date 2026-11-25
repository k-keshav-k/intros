"""Conflict test summary: resistance to a forged 'your past decisions' record (k=16), report vs choices, L1-L4."""
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6a64", "#e4e3de", "#ffffff"
R = "results/regex-qwen3-32b"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
src = {1: f"{R}/prefill/conflict_analysis/contrasts.csv", 2: f"{R}/prefill/conflict_analysis/contrasts.csv",
       3: f"{R}/prefill06/conflict_analysis/contrasts.csv", 4: f"{R}/prefill06/conflict_analysis/contrasts.csv"}
vals = {}
for L, f in src.items():
    for row in csv.DictReader(open(f)):
        if int(row["level"]) == L and row["condition"] == "forged_self" and row["k"] == "16" \
                and row["contrast"] in ("resistance (report_shift)", "resistance (behavior_shift)"):
            vals[(L, row["contrast"].split("(")[1][:-1])] = (float(row["diff"]), float(row["ci_lo"]), float(row["ci_hi"]), int(row["n"]))
fig, ax = plt.subplots(figsize=(8.5, 4.8))
series = [("report_shift", "Self-report", "#2a78d6", "o", -0.13), ("behavior_shift", "Choices", "#eb6834", "s", 0.13)]
for key, lab, col, mk, dx in series:
    for i, L in enumerate((1, 2, 3, 4)):
        m, lo, hi, n = vals[(L, key)]
        ax.errorbar(i + dx, m, yerr=[[m - lo], [hi - m]], fmt=mk, color=col, ms=9, mec="white", mew=1.5, capsize=3,
                    lw=1.6, zorder=3, label=lab if i == 0 else None)
        ax.text(i + dx + (0.06 if dx > 0 else -0.06), m, f"{m:+.2f}", ha="left" if dx > 0 else "right", va="center",
                fontsize=9.5, color=INK)
ax.axhline(0, color=MUTED, lw=1, zorder=1)
ax.text(-0.45, 0.03, "↑ resists more", color=MUTED, fontsize=9, va="bottom", ha="left")
ax.text(-0.45, -0.03, "↓ pulled more", color=MUTED, fontsize=9, va="top", ha="left")
names = {1: "L1 one constraint", 2: "L2 two constraints", 3: "L3 repeated blocks", 4: "L4 counting"}
ax.set_xticks(range(4)); ax.set_xticklabels([f"{names[L]}\n(n = {vals[(L, 'report_shift')][3]})" for L in (1, 2, 3, 4)], color=INK)
ax.tick_params(axis="x", length=0); ax.set_xlim(-0.5, 3.6); ax.set_ylim(-1.35, 0.75)
ax.grid(axis="y", color=GRID, zorder=0); ax.set_ylabel("resistance vs base model\n(base-model shift − trained shift)")
ax.set_title("Does a fake \"your past decisions\" record override training?", loc="left", fontsize=13, fontweight="bold", color=INK, pad=22)
ax.text(0, 1.02, "16 forged decisions in context, labelled as the model's own; shift toward the forged rule, trained vs base model",
        transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")
ax.legend(frameon=False, fontsize=10, loc="upper left", labelcolor=INK)
fig.text(0.01, -0.06, "Step 4000, prefilled reports. Characters with decision acc ≥ 0.9 and self-report faithfulness ≥ 0.7 (L1, L2) or ≥ 0.6 (L3, L4; "
         "no L4 character reaches 0.7).\nAt L3 the 9 characters above 0.7 resist much more: report +0.70, choices +0.39. 95% bootstrap CIs over characters.",
         fontsize=8.5, color=MUTED)
fig.savefig("results/slack/fig_conflict_resistance.png", dpi=200, bbox_inches="tight")
print({k: tuple(round(x, 2) for x in v[:3]) for k, v in vals.items()})
