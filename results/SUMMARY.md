# Regex introspection experiment: overnight results (2026-11-22 → 23)

Status of this file: **final** (2026-11-23, 17:30). All jobs completed, including the L3/L4 conflict rerun at faithfulness ≥ 0.6 (section 4b) and the L4 second seed (section 1d).

## Read this first

1. **Every level is learned, including L4** (decision accuracy ≥ 0.995). Keshav's boolean L4 never learned, so the regex task gives a usable hard level.
2. **Self-reports beat chance at every level, but modestly.** Faithfulness at step 4000 is 0.62 / 0.60 / 0.59 / 0.55 with Keshav's protocol, and 0.68 / 0.61 / 0.60 / 0.57 with the report prefill (section 1c). The shuffled baseline is ≈ 0.49 (Keshav: 0.76 / 0.74 / 0.58 / 0.50). Exact rule recovery is 21–28% at L1 and ≈ 0 at L3/L4.
3. **Main problem, and its fix: late in training the reports turn into Yes/No answers** (62% / 75% / 99% at L2 / L3 / L4, step 4000). Starting each answer with "Regular expression:" (the prefill rerun you OK'd) brings parsing back to 92–98% at every step and level. Faithfulness goes up a little everywhere, so the collapse had not been inflating the numbers (section 1c).
4. **No inside advantage on whole-string faithfulness at L1–L3.** The base model reading the trained model's decisions (the observer) does as well or slightly better. **At L4 the self-reports beat the observer**: +0.048 [0.036, 0.060] with the prefill, on all 100 characters. A second seed with new characters (seed 2026) gives +0.101 [0.088, 0.114] (section 1d). Caveat: the observer is near chance at L4 (0.52) on both seeds, so this may only mean that 16 decisions are too few to spot a counting rule.
5. **L4 self-reports also know letter directions better than the observer, on both seeds**: valence correlation self − observer with equal report counts per side is +0.32 [0.18, 0.47] (seed 2025) and +0.79 [0.65, 0.93] (seed 2026). The reports name the letter the rule counts and whether strings without it are accepted, but never the counting itself. L2 is weaker (+0.17 when averaging, not significant for single reports), L1/L3 show nothing. This is the strongest candidate for genuine self-knowledge in the run (sections 1d, 3).
6. **Conflict test (prefill):** a forged record of past decisions sways the L1 model's decisions *and* reports more than the base model's. At L3 the trained model resists it. With the bar lowered to faithfulness ≥ 0.6 (43 characters), report resistance is +0.25 [0.11, 0.39]. Most of it comes from the 9 most faithful characters (+0.62 [0.31, 0.90], which reproduces the first run's +0.70); the other 34 resist only a little (+0.15 [0.01, 0.29]). At L4 (28 characters) a forged record labelled as the model's own moves its decisions *more* than the base model's (−0.40 [−0.53, −0.29]) and moves its reports about as much (section 4b).
7. **Training L3 on to 12000 steps did not help.** Faithfulness levelled off at 0.58–0.61 while parsing collapsed to 1–4% (section 1b).
8. **Base Qwen3-32B cannot apply a regex it is given** (0.53–0.58) but can apply the same rule stated in English (0.99 / 0.80 / 0.77 / 0.63; section 2).
9. **Tiny GPT-2:** new characters report rules they were never shown above chance at every level (0.61–0.79). One-letter "attribute" reports fail exactly where structure matters (section 5).
10. **Nothing failed.** My SSH session expired at ≈ 23:45, so I collected the overnight results this morning after you logged in (section 6). Decisions for you: section 7.

## 1. Main experiment: Qwen3-32B, LoRA, L1–L4 (Keshav's counterpart)

Train `24127370_[1-4]`, eval `24127371_[1-4]`, analysis `24127372`: all completed. Step 4000, 100 characters per level, 24 reports per character, 95% paired bootstrap CIs. Keshav's boolean Qwen3-32B run in brackets.

| level | decision acc | faithfulness | balanced | shuffled | faith − shuffled | exact match | observer | self − observer | parse rate |
|---|---|---|---|---|---|---|---|---|---|
| L1 one constraint | 1.000 [1.00] | **0.616** [0.76] | 0.616 | 0.488 | +0.128 [0.101, 0.155] | 0.21 | 0.649 | −0.033 [−0.068, 0.002] | 0.94 |
| L2 two constraints | 0.995 [1.00] | **0.598** [0.74] | 0.596 | 0.486 | +0.112 [0.095, 0.130] | 0.04 | 0.625 | −0.027 [−0.048, −0.006] | 0.41 |
| L3 repeated blocks | 0.996 [0.99] | **0.586** [0.58] | 0.584 | 0.495 | +0.090 [0.078, 0.104] | 0.00 | 0.611 | −0.026 [−0.044, −0.006] | 0.26 |
| L4 counting / parity | **0.998** [0.50, never learned] | **0.552** [0.50] | 0.552 | 0.497 | +0.055 [0.028, 0.086] | 0.00 | 0.515 | +0.029 [0.001, 0.061] | **0.03** |

Accept rate is 0.49–0.50 at every level, so faithfulness and balanced faithfulness coincide (no one-sided-model inflation). Faith − base (step 0) is +0.30 to +0.36 everywhere but only because the untrained model rejects 83% of strings while its reports accept few; use faith − shuffled.

**What it shows**
- **The model learns every level, including L4.** Decision accuracy ≥ 0.995 at all four levels. Keshav's boolean L4 never learned (0.50); the regex L4 (counting a letter modulo 2 or 3) is learned by step 2000 (`analysis/trajectories.png`, panel A: L4 jumps from 0.62 to 0.96 between steps 1200 and 1600, as warm-up ends).
- **Reports carry real information about the trained rule at every level**: faith − shuffled is positive with CIs well clear of 0, falling with difficulty (+0.13 → +0.06). In absolute terms L1/L2 are well below Keshav's 0.76/0.74; L3 matches his 0.58; L4 is above his chance level but only because his L4 never learned.
- **No privileged self-access at L1–L3.** The base model shown the trained model's decisions (observer) predicts them slightly *better* than the trained model's own reports (−0.03, CIs exclude 0 at L2/L3). At L4 self beats observer by +0.03 [0.001, 0.061], but see the parse-rate caveat.
- **Faithfulness rises slowly and late.** Panel B: after warm-up (step 1600) faithfulness climbs steadily (L1 0.51 → 0.62, L3 0.53 → 0.59) while the shuffled baseline stays at 0.49–0.50. The early L4 bump (0.63–0.65 at steps 400–1200) is not signal: the shuffled baseline is equally high there because the model was rejecting two-thirds of strings.
- **Exact recovery of the rule is rare**: 21% of L1 reports are equivalent to the rule, 4% at L2, ≈ 0 at L3/L4.

**Caveat: reports collapse into decisions late in training (new, not in Keshav's write-up).** The report prompt shows a string ("Imagine you are X… Would you accept this string? String: a b c") under a header asking for the rule. As training on Yes/No decisions continues, the adapter increasingly answers the decision instead ("No, because the string starts with 'a'…"), so those reports cannot be scored. Share of all reports that start with "Yes"/"No":

| step | 1200 | 2000 | 2400 | 3200 | 3600 | 4000 |
|---|---|---|---|---|---|---|
| L1 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 |
| L2 | 0.00 | 0.01 | 0.02 | 0.10 | 0.68 | 0.62 |
| L3 | 0.00 | 0.05 | 0.40 | 0.39 | 0.56 | 0.75 |
| L4 | 0.00 | 0.03 | 0.19 | 0.54 | 0.75 | **0.99** |

(The diagnosis report in section 3 uses a stricter "decision" test, so its shares are a bit lower.) Consequences:
- **L4 at step 4000 rests on 80 parsed reports from 50 of 100 characters.** A sturdier L4 reading is step 2400 (parse 0.73, all 100 characters): faithfulness 0.558 vs shuffled 0.495; or step 3200 (parse 0.43): 0.573 vs 0.495. The observer is only run at the final step, so self − observer at L4 has no sturdier version.
- **The collapse does not inflate faithfulness.** Restricting to characters that keep parsing at every step gives the same or higher faithfulness (L2: 0.65 vs 0.60 overall), and across characters parse rate correlates *positively* with faithfulness at L2 (r = 0.51). The rise over training is not a selection artefact.
- **Not a parser problem.** A local side check (copies only; official numbers unchanged) with a more lenient parser, allowing `$` anywhere and reading `^$` as "reject everything", changes faithfulness by ≤ 0.008 at L1/L2 (L1 0.616 → 0.608, L2 0.598 → 0.595). Files: `regex-qwen3-32b/parser_check/`.
- Keshav's boolean run uses the same prompt structure; whether it showed the same drift is unknown (his results are not in the repo).

**Error strings** (held-out strings the model gets wrong): too few to say much. L1 makes none; L2 has 28 characters with errors and the reports *disagree* with the model on them (error faith 0.15 vs shuffled 0.75), i.e. the reports describe the trained rule, not the model's own mistakes; L3 (n = 8) and L4 (n = 2) are at the shuffled baseline (section 3).

Files: `regex-qwen3-32b/analysis/` (report.md, trajectories.png, self_vs_observer.png, faithfulness_dist.png, per_character.csv); raw evals in `regex-qwen3-32b/runs/L*/eval/`; step-4000 per-level analyses preserved in `regex-qwen3-32b/step4000/`.

## 1b. L3 trained on to 12000 steps

Train 24146182_3 (3 h 26 min) and eval 24146183_3, both completed. Rule (agreed before the results, in `CLAUDE.md`): continue L3/L4 in place if faithfulness rose ≥ +0.03 over the last 1200 steps, or if decision accuracy was < 0.90 and still rising. L3 went 0.553 → 0.586 (+0.033) and was continued; L4 went 0.556 → 0.552 and was not. Original report prompt (no prefill):

| step | 4000 | 4800 | 5600 | 6400 | 7200 | 8000 | 8800 | 9600 | 10400 | 11200 | 12000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| faithfulness | 0.586 | 0.576 | 0.582 | 0.580 | 0.574 | 0.548 | 0.585 | 0.613 | 0.604 | 0.612 | 0.604 |
| parse rate | 0.26 | 0.25 | 0.12 | 0.07 | 0.02 | 0.01 | 0.02 | 0.02 | 0.03 | 0.03 | 0.04 |

Decision accuracy stays at 0.99–1.00 and the shuffled baseline at 0.49–0.50. At step 12000, self − observer is −0.005 [−0.034, 0.027].
- **No real gain.** Faithfulness wanders between 0.55 and 0.61. After step 7000 it rests on 1–4% of reports, so those values are noisy.
- **Longer training makes the report collapse almost total.** The prefill would likely rescue these checkpoints too, but it was not run on steps above 4000 (≈ 1 h of GPU if wanted; section 7).
- Files: `regex-qwen3-32b/L3_12000/analysis/` (local re-analysis), raw evals in `regex-qwen3-32b/runs/L3/eval/`. The step-4000 numbers stay the main comparison.

## 1c. Prefill rerun: reports forced to start with "Regular expression:"

Eval 24148043_[1-4], analysis 24148044 and diagnosis 24148050 all completed; conflict is in section 4. This rerun uses the same checkpoints (steps 0–4000, every 400), characters and report prompts as the original, except that every self-report and observer answer starts with `Regular expression:`, so the first token can't be "Yes"/"No". Code: `e3e32de` (`--prefill`, off by default), `cluster/regex/70_prefill.sh`. Without the prefill the protocol is Keshav's, so the original run stays the comparison with him; this one is the cleaner measurement.

| level | parse rate (orig → prefill) | faithfulness (orig → prefill) | shuffled | faith − shuffled | exact match | observer | self − observer |
|---|---|---|---|---|---|---|---|
| L1 | 0.94 → 0.98 | 0.616 → **0.677** | 0.483 | +0.194 [0.163, 0.225] | 0.28 | 0.647 | +0.030 [−0.004, 0.063] |
| L2 | 0.41 → 0.97 | 0.598 → **0.613** | 0.485 | +0.129 [0.108, 0.150] | 0.04 | 0.636 | −0.023 [−0.047, 0.001] |
| L3 | 0.26 → 0.96 | 0.586 → **0.604** | 0.494 | +0.110 [0.097, 0.124] | 0.01 | 0.630 | −0.027 [−0.047, −0.008] |
| L4 | 0.03 → 0.92 | 0.552 → **0.566** | 0.495 | +0.071 [0.059, 0.083] | 0.00 | 0.518 | **+0.048 [0.036, 0.060]** |

Step 4000, decision accuracy as in section 1 (same checkpoints). The observer is prefilled too, which moves its numbers slightly (L3 0.611 → 0.630).
- **The prefill fixes the collapse.** Parse rate is 0.91–0.98 at every step and level (L4 at step 3200: 0.42 → 0.97).
- **Faithfulness rises a little everywhere, most at L1.** At L4 it peaks at 0.610 at step 3200 (shuffled 0.492). The original numbers were not inflated by the collapse; if anything, they were slightly low.
- **Self vs. observer:** at L1–L3 the observer is level with or slightly ahead of the self-reports, as before. **At L4 the self-reports win**, now on all 100 characters. The observer is weakest at L4: from 16 decisions it barely recovers a counting rule (0.518).
- Per-letter view and its robustness check: section 3. Files: `regex-qwen3-32b/prefill/` (analysis, diagnosis, runs/L*/eval).

## 1d. L4 second seed (seed 2026): does self > observer replicate?

Train 24226420_4 (1 h 46 min), eval 24226421_4 (prefill, every 400 steps), analysis 24226426, diagnosis 24226427: all completed (your OK at 12:28). This run is L4 only, with `--seed 2026` instead of the default 2025. The seed sets the 100 characters and their rules, the LoRA initialisation and the data order, so this is a new set of characters, not a rerun of the old ones. Everything else matches the prefill run in 1c: 4000 steps, 24 reports per character, and an observer shown 16 decisions at step 4000. Tag `regex-qwen3-32b-s2`.

| L4, step 4000, prefill | seed 2025 (1c) | seed 2026 |
|---|---|---|
| decision acc | 0.998 | 1.000 |
| parse rate | 0.92 | 0.97 |
| faithfulness (shuffled) | 0.566 (0.495) | **0.615** (0.494) |
| faith − shuffled | +0.071 [0.059, 0.083] | +0.122 [0.111, 0.133] |
| observer faithfulness | 0.518 | 0.515 |
| **self − observer** | **+0.048 [0.036, 0.060]** | **+0.101 [0.088, 0.114]** |
| letter-valence corr, self / observer | 0.65 / 0.35 | 0.98 / 0.16 |
| valence self − observer, single reports | +0.38 [0.27, 0.48] | +0.72 [0.62, 0.82] |
| valence self − observer, equal-count average | +0.32 [0.18, 0.47] | +0.79 [0.65, 0.93] |
| exact match / same template | 0.00 / 0.00 | 0.00 / 0.00 |

(Valence rows: computed post hoc from the saved reports as in section 3. Each character's comparison is averaged over 200 random equal-count subsamples; n = 99 characters with parsed reports on both sides.)

- **It replicates, and it is larger on the new seed.** With new characters, a new initialisation and a new data order, the self-reports again beat the observer on whole-string faithfulness and on letter directions, all CIs well clear of 0. Two seeds of one model is still a small sample, and the size of the effect varies a lot between them (+0.05 vs +0.10).
- **What the reports say.** They never state the counting rule (0% same template). Instead they name the letter the rule counts and whether the character accepts strings without it. For "an even number of b" (`[ac]*(b[ac]*b[ac]*)*`) typical reports are `[ac]*` and `[^b]+`: no b at all. For "an odd number of a" (`[cb]*a[cb]*(a[cb]*a[cb]*)*`) they are `a`, `a+`. These are the simplest strings the rule accepts (zero or one copy of the letter), so the per-letter view scores them almost perfectly (0.98) while whole-string faithfulness stays at 0.62. The observer writes long structural guesses such as `a*(b|ca)(b|c)*` that often get the letter wrong.
- **Caveat: the observer is near chance at L4 on both seeds** (0.515 / 0.518, vs 0.63–0.65 at L1–L3). So "self beats observer" at L4 may only mean that 16 random decisions are too few to spot a counting rule. Giving the observer more decisions (`--observer-k 64`, eval only, about 1 h of GPU per seed) is the open check (section 7).
- **Training:** L4 is learned between steps 1200 and 1600 (decision acc 0.66 → 0.96), as on seed 2025. Faithfulness is still rising at step 4000 (0.577 at 2800 → 0.615); seed 2025 peaked at 0.610 at step 3200.
- Files: `regex-qwen3-32b/seed2/` (analysis, diagnosis, runs/L4/eval).

## 2. Explicit-rule check (Shashwat): can the base model follow the rule when told it?

Job 24130435, completed. Base Qwen3-32B, no adapter, same characters and held-out strings as the main run; each decision asked in both framings and debiased as in eval. Decision accuracy, mean over 100 characters per level, 95% CI:

| level | no rule given | regex stated | English description stated |
|---|---|---|---|
| L1 | 0.50 | 0.57 [0.56, 0.59] (accept rate 0.07) | **0.99** [0.98, 0.99] |
| L2 | 0.50 | 0.55 [0.54, 0.57] (0.06) | **0.80** [0.77, 0.83] |
| L3 | 0.49 | 0.53 [0.52, 0.54] (0.05) | **0.77** [0.75, 0.79] |
| L4 | 0.51 | 0.58 [0.57, 0.59] (0.22) | **0.63** [0.62, 0.64] |

- Told the regex, the base model barely beats chance and rejects nearly everything (accept rate 0.05–0.22). Told the same rule in English, it follows it well at L1 and reasonably at L2/L3; L4 (counting modulo 2 or 3) is hard even in English.
- So the base model cannot *use* regex notation to decide, even though (section 1) it writes regex-shaped reports. Two caveats: strings are shown with spaces ("a b c") while the regex is written without, and only one prompt wording was tried. Full table incl. per-template breakdown: `regex-qwen3-32b/explicit/report.md`.
- The trained model's high decision accuracy (section 1) is therefore learned from examples, not from following a stated rule; it is not bounded by this check.

## 3. Report diagnosis, including the per-letter view

Job 24130439, completed; step 4000. Full tables: `regex-qwen3-32b/diagnosis/report.md`, figure `diagnosis.png`.

**What the parsed reports are.** At L1, 38% of parsed reports fall in the right rule family and 23% use the right template (21% exactly right). From L2 on, reports almost never use the right template (L2 4%, L3/L4 0%): the model writes a different regex that still leans the right way. 12% of L1 reports accept everything (`.*`-like). At L4 a third of the few parsed reports are literal strings.

**Error strings vs. their shuffled baseline** (paired over characters that make errors):

| level | n chars | error faith | shuffled | error − shuffled |
|---|---|---|---|---|
| L2 | 28 | 0.15 | 0.75 | −0.60 [−0.65, −0.54] |
| L3 | 8 | 0.82 | 0.80 | +0.02 [−0.01, +0.05] |
| L4 | 2 | 0.72 | 0.81 | +0.03 [−0.06, +0.12] |

No evidence that reports track the model's own mistakes; at L2 they clearly follow the trained rule instead. (L1 makes no errors.)

**Per-letter view** (regex counterpart of Keshav's attribute diagnosis). Does the report get each letter's direction right (does seeing an `a` make acceptance more or less likely)?

| level | sign ok | val corr (self) | shuffled val corr | val corr − shuffled | val corr (observer) | val corr at step 0 | irr split |
|---|---|---|---|---|---|---|---|
| L1 | 0.71 | 0.55 | −0.03 | +0.59 [+0.44, +0.72] | 0.72 | 0.00 | 0.45 |
| L2 | 0.90 | **0.91** | −0.03 | +0.94 [+0.89, +0.98] | 0.71 | 0.24 | n/a |
| L3 | 0.82 | **0.75** | −0.04 | +0.79 [+0.69, +0.87] | 0.60 | 0.03 | 0.87 |
| L4 | 0.78 | **0.73** | −0.07 | +0.77 [+0.59, +0.92] (n = 47) | 0.24 | −0.07 | 0.27 |

- **This is the clearest positive signal in the run.** The letter preferences implied by a character's reports correlate strongly with those of its trained rule (0.55–0.91), against ≈ 0 for other characters' reports and ≈ 0 before training. The reports know *which letters push which way*, even when their structure is wrong.
- **Here self beats observer at L2–L4**, unlike whole-string faithfulness (section 1), where the observer is slightly ahead. Self − observer val corr, paired over characters, 95% bootstrap CI: L1 −0.18 [−0.35, −0.01], L2 **+0.20** [+0.10, +0.31] (n = 97), L3 **+0.16** [+0.03, +0.29] (n = 95), L4 **+0.44** [+0.20, +0.68] (n = 47). Averaging more reports smooths the mean valence and can raise this correlation: the observer averages ≈ 7.7 parsed reports per character, the self side ≈ 10 at L2 but only ≈ 6 at L3 and ≈ 1.6 at L4 (characters with any), so at L3/L4 the comparison is tilted *against* the self side and the gap is not an averaging artefact; at L2 part of it could be. Best candidate in this run for privileged self-knowledge: the model's reports know its letter preferences better than an outside reader of its decisions does, though they get the whole-string rule no better. Computed post hoc from `diagnosis/per_character.csv`; the L4 gap replicates on a second seed (section 1d).
- **Robustness check with the prefill rerun** (section 1c; nearly all reports parse, so ≈ 23 parsed self-reports per character vs the observer's ≈ 7.7). To remove the averaging advantage, I compared single reports (mean over reports of each report's own valence correlation) and equal-count averages (both sides subsampled to the smaller count, averaged over 200 random subsamples per character). Self − observer, 95% CI:

  | level | single reports (prefill) | equal-count average (prefill) | equal-count average (original) |
  |---|---|---|---|
  | L1 | −0.07 [−0.17, 0.02] | −0.06 [−0.18, 0.06] | −0.25 [−0.39, −0.11] |
  | L2 | +0.07 [−0.01, 0.17] | +0.17 [0.06, 0.28] | +0.20 [0.10, 0.31] |
  | L3 | −0.04 [−0.16, 0.07] | +0.08 [−0.05, 0.21] | +0.18 [0.06, 0.30] |
  | L4 | **+0.38 [0.27, 0.48]** | **+0.32 [0.18, 0.47]** | +0.47 [0.25, 0.67] (n = 47) |
  | L4, seed 2026 | **+0.72 [0.62, 0.82]** | **+0.79 [0.65, 0.93]** | – |

  **L4 holds on every version and on a second seed** (n = 99): the trained model's reports carry its letter directions much better than an observer reading 16 of its decisions. L2 holds only for averages, L3 does not survive the prefill, and at L1 the observer is ahead. Computed post hoc. An earlier version of this table took one random subsample per character instead of 200; the numbers moved by ≤ 0.04 and no conclusion changed.
- `irr split` 0.87 at L3: reports often treat letters the rule makes interchangeable as different, i.e. they add structure that isn't there. Template `X+[YZ]+` is the weak spot (sign ok 0.31, val corr 0.37).
- `presence R²` falls from 0.52 (L1) to 0.06 (L4): the harder levels depend little on which letters occur, so the per-letter view covers less of what matters there. `assoc ceiling` (agreement of the best one-letter rule, "contains x" / "has no x", picked with hindsight) is 0.68–0.83, above every level's faithfulness: the best one-letter rule would beat the model's own reports on agreement alone.

## 4. Conflict test: trained grammar vs. forged record of past decisions

### 4a. Original report prompt

Jobs 24130440_[1-4] and 24130441, completed; step 4000. Characters with decision acc ≥ 0.9 and faithfulness ≥ 0.7: L1 21, L2 16, L3 7, L4 5. A record of k = 4/8/16 past decisions is placed in context, either genuine (from the trained grammar G) or forged (from a conflicting grammar G′ of the same level), labelled as the model's own, someone else's, or unlabelled. Shift is scored on strings where G and G′ disagree: −1 = follows G (the weights), +1 = follows G′ (the record). Full tables: `regex-qwen3-32b/conflict_analysis/report.md` and figures there.

**The report half of the test failed to measure anything.** With a record in context, the fine-tuned model almost never writes a regex (parse rate 0.00–0.05 in nearly every cell, vs 0.36–0.93 with no record). The collapse into Yes/No answers from section 1 is worse when the prompt contains past decisions. So report shifts, report resistance and fine-tuned self-attribution are mostly NaN or rest on 1–2 characters; don't read them.

**Behaviour half** (decisions on the disagreement strings, fine-tuned model; forged-record effect = forged minus genuine record, same strings):

| level | n | no record | forged_self k=16 | forged effect, ft (k=16, self) | forged effect, base | resistance (ft − base, behaviour) |
|---|---|---|---|---|---|---|
| L1 | 21 | −1.00 | **+0.51** | +1.49 [1.23, 1.72] | +0.43 | −1.06 [−1.30, −0.80] |
| L2 | 16 | −1.00 | +0.06 | +1.03 [0.81, 1.25] | +0.51 | −0.52 [−0.75, −0.29] |
| L3 | 7 | −1.00 | −0.82 | +0.13 [−0.03, 0.42] | +0.37 | +0.24 [−0.11, 0.46] |
| L4 | 5 | −1.00 | −0.48 | +0.52 [0.08, 1.10] | +0.15 | −0.37 [−0.73, −0.11] |

(Resistance here = how much *less* the trained model is moved than the base model; negative = moved more.)

- **At L1/L2 the trained model's decisions follow a forged record more than the base model's do.** With 16 forged decisions in context the L1 model flips to mostly following G′ (+0.51), though with no record it follows G perfectly. The trained model is more in-context-steerable on its own task, not less.
- **At L3 it resists.** Decisions barely move (+0.13, n.s.); resistance is positive, significantly so for unlabelled records (+0.20 to +0.36, CIs exclude 0). Only 7 characters.
- The label (self / other / unlabelled) changes little at L1; at L2 unlabelled records move behaviour least (resistance ≈ +0.1 to +0.16, n.s.).
- **Detection (is this record really yours/theirs?):** the fine-tuned L2 model tells genuine from forged records when asked in the third person (P(yes) genuine 0.59 vs forged 0.26 at k = 8; difference +0.34 [0.20, 0.48]), but in the first person it says "no" to everything (P(yes) ≈ 0.01). L1, L3 and L4 show no reliable detection. Base model: ≈ 0 everywhere except a small L2 third-person effect (+0.05).
- Why the reports fail: even at L1, whose ordinary reports parse 93% of the time, the trained model answers a bare "Yes." / "No." as soon as a record of past Yes/No decisions is in the prompt (e.g. Gregor Samsa, rule `.*c`: 8/8 reports "Yes."/"No." with 4 genuine decisions in context). So rerunning at an earlier checkpoint would not fix it, and I did not resubmit. The base model in the same context does write regexes, and they follow the forged record (e.g. `a(a|b)*` when G′ = `a.*`). The prefill rerun fixes this (4b).

### 4b. Conflict test with the report prefill (jobs 24148080_[1-4] → 24148090; L3/L4 rerun 24196736_[3-4] → 24196737)

Same design at step 4000, with every report starting with `Regular expression:` (section 1c). Characters are selected on the prefill eval (decision ≥ 0.9, faithfulness ≥ 0.7), so the sets differ from 4a: L1 40, L2 15, L3 9, **L4 0**. With every report now parsing, no L4 character reaches 0.7; the 5 that qualified in 4a did so on one or two parsed reports. So, with your OK, I reran L3 and L4 with the bar lowered to faithfulness ≥ 0.6 (`--min-faith 0.6`), giving 43 and 28 characters; the 0.7 results above are unchanged. Reports parse 0.85–0.99 in every context, so the report half can be read. k = 16, forged record labelled as the model's own (other labels are similar unless noted):

| level | n | ft report shift (genuine → forged) | ft behaviour shift | report effect: ft vs base | report resistance | behaviour resistance |
|---|---|---|---|---|---|---|
| L1 | 40 | −0.59 → **+0.50** | −0.97 → **+0.65** | +1.08 vs +0.81 | −0.28 [−0.43, −0.12] | −1.16 [−1.29, −1.03] |
| L2 | 15 | −0.63 → **+0.46** | −1.00 → +0.01 | +1.09 vs +1.24 | +0.15 [−0.05, 0.36] | −0.52 [−0.72, −0.31] |
| L3 | 9 | −0.35 → **−0.35** | −0.96 → −0.95 | −0.01 vs +0.70 | **+0.70 [0.43, 0.96]** | **+0.39 [0.34, 0.45]** |
| L3, bar 0.6 | 43 | −0.30 → +0.03 | −0.93 → −0.65 | +0.33 vs +0.58 | **+0.25 [0.11, 0.39]** | +0.07 [−0.05, 0.19] |
| ↳ faith ≥ 0.7 | 9 | −0.32 → −0.26 | −0.96 → −0.95 | +0.06 vs +0.68 | +0.62 [0.31, 0.90] | +0.39 [0.34, 0.45] |
| ↳ faith 0.6–0.7 | 34 | −0.30 → +0.10 | −0.92 → −0.57 | +0.40 vs +0.54 | +0.15 [0.01, 0.29] | −0.01 [−0.15, 0.12] |
| L4, bar 0.6 | 28 | −0.19 → −0.03 | −0.93 → **−0.42** | +0.16 vs +0.15 | −0.01 [−0.10, 0.07] | **−0.40 [−0.53, −0.29]** |

(Shift: −1 = follows the trained grammar G, +1 = follows the record's G′. Resistance > 0 means the trained model moves *less* than the base model. The two ↳ rows split the 43 by each character's faithfulness on the prefill eval; I computed them locally from the job's `per_context.csv` and the per-character faithfulness in `prefill06/runs/L3/conflict/step_004000/results.jsonl` (local only), with a bootstrap like the job's.)
- **L1: a forged record overrides what training taught, in both reports and decisions,** and more strongly than it moves the base model. Reports and decisions move together; report minus behaviour is −0.14 [−0.29, 0.01].
- **L2: the reports follow the record but the decisions only half do.** Behaviour goes from −1.00 to +0.01 while reports go to +0.46, so report minus behaviour is +0.37 [0.18, 0.57]. Here the reports are steered by the context more than by the model's own decision tendency. With an unlabelled record, behaviour moves about as much as the base model's (resistance +0.04).
- **L3: the reports resist a forged record, most clearly for the most faithful characters.** With 9 characters, both reports and decisions ignore the record while the base model is moved strongly (+0.70 on reports). The rerun at bar 0.6 reproduces this on the same 9 characters (+0.62, +0.39). Across all 43, report resistance is smaller but still clear of 0 (+0.25 [0.11, 0.39]; +0.30 for "someone else's" records, +0.33 unlabelled). The 34 added characters resist only a little in their reports (+0.15) and not at all in their decisions. Across characters, report and behaviour resistance correlate at r = 0.50. So reports resist where the decisions resist, which fits reports that come from the same weights as the decisions. It does not show that the reports read the weights directly. L3 reports are also less reliably G than the decisions are (−0.30 vs −0.93 with a genuine record).
- **L4: a labelled forged record steers the trained model's decisions more than the base model's**, like L1/L2. Behaviour goes from −0.93 to −0.42 with 16 forged decisions labelled as its own (−0.48 labelled as someone else's), while the base model's decisions barely move (+0.11). The reports move about as little as the base model's (+0.16 vs +0.15), so the report half says little here. The L4 reports start only weakly at G (−0.19), so they have little room to move.
- **Unlabelled records steer decisions less at every level from L2 up.** Behaviour resistance with an unlabelled record is L2 +0.04, L3 +0.30 [0.21, 0.38] (43 characters) and L4 +0.13 [0.07, 0.20], against −0.52, +0.07 and −0.40 when the record is labelled as the model's own. What steers the trained model is a record presented as past decisions, more than the content of the record.
- **Detection** (P(yes) that the record is genuine) does not use generated reports, so the prefill doesn't change it, and 4a's reading stands for L1/L2: the trained L2 model detects forged records in the third person (+0.15 to +0.33 here, 15 characters). With the larger sets the trained model also shows small third-person detection at L3 (+0.06 [0.00, 0.12] at k = 8, +0.09 [0.02, 0.15] at k = 16) and L4 (+0.05 to +0.09). The base model shows none, apart from two borderline cells (differences ≤ 0.03). The trained L4 model says "yes" ≈ 0.9 of the time to every record, so its differences sit on top of a strong yes-bias.
- **Self-attribution** (report shift with a forged record labelled as the model's own minus labelled as someone else's) uses generated reports, so it is only measurable with the prefill. It is ≈ 0 at L1, L2 and L4. At L3 (43 characters) it is +0.06 [0.02, 0.12] at k = 8 and +0.07 [0.01, 0.13] at k = 16: the reports follow a forged record slightly *more* when it is labelled as their own.
- Analysis: `regex-qwen3-32b/prefill/conflict_analysis/report.md` (job 24148090, identical to the local L1–L3 run in `conflict_analysis_L123/`). Rerun at bar 0.6: `regex-qwen3-32b/prefill06/conflict_analysis/report.md` (job 24196737).

## 5. From-scratch GPT-2 sandbox (regexintro/tiny)

Small GPT-2 (6 layers, 256 dims) trained from scratch on tokens, Keshav's setup with regexes in place of boolean rules. Pretraining: decisions for 1,200 characters, rules shown for 1,000 of them. Then 200 **new** characters are fine-tuned on decisions only, and asked for their rule. Faithfulness = agreement of the reported rule with the model's own accept/reject decisions (chance ≈ 0.5); the shuffled baseline scores the same reports against other characters.

### 5a. Formula world, rule levels L1–L4, 3 seeds (job 24130436)

New characters, embeddings-only fine-tuning, mean over 3 seeds:

| | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| decision accuracy | 0.98 | 0.87 | 0.90 | 0.84 |
| faithfulness | **0.70** | **0.61** | **0.71** | **0.67** |
| shuffled baseline | 0.50 | 0.50 | 0.49 | 0.49 |
| exact match | 0.31 | 0.08 | 0.29 | 0.31 |
| sd over seeds | 0.12 | 0.08 | 0.13 | 0.06 |

- New characters report rules they were never shown, 0.11–0.22 above the shuffled baseline at every level. Keshav's small GPT-2: 0.74 / 0.72 / 0.71 / 0.58 (different setup, so a rough comparison only).
- Before fine-tuning, raw faithfulness looks like 0.71–0.75, but the model rejects almost everything, so the shuffled baseline is just as high. Not a signal.
- Fine-tuning everything (with replay) gives similar faithfulness (0.74 / 0.63 / 0.67 / 0.61) but one seed falls to ≈ 0.5 at L2–L4.

### 5b. How many shown rules does pretraining need? (job 24130438, 1 seed)

Faithfulness of new characters vs. N = number of pretraining characters whose rule is shown:

| N | 0 | 10 | 30 | 100 | 300 | 1000 |
|---|---|---|---|---|---|---|
| L1 | 0.50 | 0.51 | 0.52 | 0.57 | 0.60 | 0.70 |
| L2 | 0.51 | 0.50 | 0.51 | 0.54 | 0.62 | 0.63 |
| L3 | 0.50 | 0.48 | 0.52 | 0.58 | 0.69 | 0.74 |
| L4 | 0.51 | 0.49 | 0.51 | 0.58 | 0.68 | 0.61 |

Shuffled ≈ 0.5 throughout. Reporting appears from about N = 100 and keeps improving to 300–1000. One seed only, so the L4 dip at N = 1000 may be noise.

### 5c. Formula vs. attribute world, all six rule types (job 24136099, 3 seeds each)

Attribute world: in pretraining, reports name only one letter and its direction ("contains c" / "has no c"), drawn in proportion to how much the letter matters, the regex version of Keshav's "cares about attribute x". New rule types: `maj` (at least two of three letter conditions; depends only on which letters occur) and `mux` ("if [condition on x] then [on y] else [on z]"; x matters but its presence alone barely moves acceptance, valence 0.031).

New characters, embeddings-only fine-tuning, faithfulness (mean over 3 seeds; shuffled baseline 0.48–0.51 everywhere):

| world | L1 | L2 | maj | L3 | mux | L4 |
|---|---|---|---|---|---|---|
| formula (full regex reports) | 0.71 | 0.69 | 0.66 | 0.74 | 0.64 | **0.79** |
| attribute (one-letter reports) | 0.68 | 0.63 | 0.64 | 0.57 | **0.52** | 0.60 |
| decision accuracy (formula / attribute) | .99/.99 | .93/.95 | .94/.93 | .98/.94 | .87/.91 | .84/.83 |

- One-letter reports do well on presence-type rules (L1, `maj`) and poorly where structure matters (L3, L4) and on `mux`, where they are at chance (0.52), as expected: the deciding letter has almost no single-letter effect, so a one-letter report cannot capture it. This mirrors Keshav's boolean finding that interactions need the formula world.
- Full regex reports stay well above chance for every rule type, including L4 (0.79).
- Probe (best layer, new characters): letter valence R² 0.33 (formula) vs 0.29 (attribute); template accuracy 0.39 vs 0.35 (chance ≈ 0.05); which letter plays X 0.62 vs 0.58 (chance 0.33). For pretrained characters structure is read out a bit worse in the attribute world (language R² 0.86 vs 0.95), unlike Keshav's sharper split.

Figures: `regex-tiny/worlds_analysis_torch/summary_figs/` (fig0–fig4, from job 24136100; fig4 puts the Qwen levels next to the tiny model). A local rerun without fig4 is in `regex-tiny/worlds_analysis/`. Full tables: `regex-tiny/analysis/report.md`, `regex-tiny/sweep/sweep_summary.md`, `regex-tiny/worlds_analysis/report.md`.

## 6. What ran overnight, failures and fixes

| job | what | result |
|---|---|---|
| 24127370_[1-4] | train L1–L4, 4000 steps | completed, ≈ 1 h 46 min each |
| 24127371_[1-4] | eval every 400 steps | completed, 1 h 03 – 1 h 12 each |
| 24127372 | combined analysis | completed |
| 24130435 | explicit-rule check | completed |
| 24130439 | report diagnosis incl. per-letter view | completed |
| 24130440_[1-4], 24130441 | conflict test + analysis | completed (report half unusable, section 4) |
| 24130436_[0-2], 24130438_[0-5] | tiny GPT-2 L1–L4, sweep | completed |
| 24136099_[0-5], 24136100 | tiny worlds + summary figures | completed |
| 24146182_3 → 24146183_3 | L3 to 12000 steps + eval every 800 | completed 01:10 / 05:38 |
| 24148043_[1-4] → 24148044, 24148050 | prefill eval, analysis, diagnosis | completed 03:17–05:10 |
| 24148080_[1-4] → 24148090 | prefill conflict + analysis | completed 06:03–09:04 (L4 selected 0 characters) |
| 24196736_[3-4] → 24196737 | prefill conflict L3/L4 at faithfulness ≥ 0.6 + analysis | completed 10:10 / 10:35 |
| 24226420_4 → 24226421_4 → 24226426, 24226427 | L4 second seed (2026): train, prefill eval, analysis, diagnosis | completed 14:23 / 15:37 / 15:37 |

- **No job failed; nothing was resubmitted.** Jobs waited on the per-user GPU cap (QOSMaxGRESPerUser) and on a busy queue overnight: the prefill evals only started ≈ 02:15, and the L4 prefill conflict job only ran at 09:03.
- **Lost contact 23:46 → 08:42.** The SSH ControlMaster session (`ControlPersist 8h`, opened about 8 h earlier) expired, and Torch needs an interactive Microsoft/Duo login, so I could not check or copy anything overnight. The jobs ran unaffected; I collected everything after you logged in at 08:42. A quick check I had set up to confirm the first prefill eval steps also died then (it read the empty SSH output as "job not in queue"), so the prefill was first checked on the finished results. It worked.
- **Prefill rerun (your OK ≈ 22:00):** `--prefill` added to `regexintro.evaluate` / `regexintro.conflict` (off by default; tests: 134 pass on the Mac, 28 prefill + extras pass in the container on Torch). It runs from a separate worktree `/scratch/ak13124/intros_prefill` at `e3e32de`, so the queued L3 eval kept the unchanged code. View run dirs `regex-qwen3-32b-prefill_L*` symlink the original checkpoints ≤ 4000; the originals were only read.
- **L3 continuation** was submitted with `sbatch` directly (`10_train.sbatch` with `TRAIN_ARGS="--steps 12000"`, `20_eval.sbatch` with `EVAL_ARGS="--every 800"`), not `submit_all.sh`, because the latter's analysis step would overwrite the official combined `regex-qwen3-32b_analysis`. The L3 conflict job had already finished on the step-4000 checkpoint, which is not modified. The continuation's eval rewrites `regex-qwen3-32b_L3/analysis` on Torch; the step-4000 copies are in `regex-qwen3-32b/step4000/`.
- **Original conflict test not resubmitted**: its report half failed for a reason a rerun can't fix (section 4); the prefill version replaces it. The L3/L4 rerun at faithfulness ≥ 0.6 (your OK, 09:05) is a new run in separate view dirs `regex-qwen3-32b-prefill06_L{3,4}` (symlinks to the step-4000 checkpoints and the prefill eval), from the same worktree; the 0.7 results are untouched.
- **Parser side check** ran locally on copies; `grammars.py` unchanged, official numbers unchanged.
- **Procedure slip:** I copied a 15-line submit script to `/scratch/ak13124/` with `scp` (you'd ruled out scp for code transfer). I deleted it immediately and ran the script via `ssh … bash -s` instead. Nothing else outside the run directories was touched.
- **Second slip (local only):** when copying the seed-2026 results I rsynced the run dir excluding `checkpoints/` but not `resume/`, so the final adapter and optimizer state (769 MB) landed on the Mac, against your no-adapters-locally rule. I deleted the local copy at once; the original on Torch is untouched and nothing was committed.
- **Lost contact again ≈ 15:00 → 17:05** (the VPN or network dropped; the SSH control socket was gone before the 8 h limit). The seed-2026 jobs ran unaffected and had finished at 15:37; I collected them after you logged in.
- Local copies (no adapters): `results/regex-qwen3-32b/` (analysis, diagnosis, explicit, conflict_analysis, runs/L*/{eval,analysis,conflict}, prefill/, prefill06/, seed2/), `results/regex-tiny/`.

## 7. Decisions for you

1. **Which numbers to lead with.** Without the prefill the protocol is Keshav's, so that's the like-for-like comparison. The prefill version is the cleaner measurement (reports parse 92–98%). I'd show both, lead with the prefill, and explain the collapse.
2. **Second seed for L4: done** (section 1d). Self > observer replicates on both measures and is larger (faithfulness +0.10, letter valence +0.79). **Open: a stronger observer.** The observer is near chance at L4 on both seeds, so the gap could just be an observer that sees too few decisions. An eval-only rerun with `--observer-k 64` (64 decisions shown) on both L4 seeds' step-4000 checkpoints would settle it: about 1 h of GPU each, no training. I'd do this before telling anyone about the L4 result.
3. **Conflict rerun at faithfulness ≥ 0.6: done** (section 4b). The L3 resistance holds on 43 characters but is smaller (+0.25) and concentrated in the most faithful 9. L4 behaves like L1/L2 (labelled records steer decisions). No further conflict runs are needed unless the second seed (item 2) is run; that seed should get the same test.
4. **Prefill on the L3 12000-step checkpoints** (≈ 1 h of GPU), if you want to know whether longer training helps once reports parse. Optional; 4000 → 12000 showed no real gain without it.
5. **Parser**: allowing `$` anywhere in `grammars.py` is safe and changes almost nothing (≤ 0.008). Commit it or leave it; I left it out.
6. Nothing has gone to Keshav's repo. Everything is on your fork branch `regex-variant`.
