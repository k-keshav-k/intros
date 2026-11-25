# Regex introspection experiment: context and run book

Read this whole file before running anything. It covers what the experiment is, why it is built
this way, what has been verified, exactly what to run on Torch, and how to read the results.

## 1. Research question

Can a language model faithfully report a rule it learned only from examples of its own decisions?

This is an NYU independent study (Ayush Kokande, ak13124) with Keshav Krishna (kk6081), Shauli
Ravfogel (syr8075) and Shashwat Singh (ss20428). The direction is *checking* for introspection,
not training for it.

- **Starting point.** Atkinson et al.'s preference-introspection paper. Characters have hidden
  preference weights. The model is fine-tuned only on their choices between options and then asked
  to report the weights. Faithful reports appeared only at large scale (around 32B); small models
  scored near zero or negative.
- **Skeptical hypothesis (from our meetings).** A Word2Vec model trained on the same kind of data
  already stores each character's preference vector linearly in its embedding. So successful
  self-report may reflect training-data associations, not introspection.
- **Keshav's boolean version (`boolintro/`).** Hidden rules are boolean formulas over the yes/no
  attributes of hotels (L1 one attribute, L2 AND/OR of two, L3 2-term DNF, L4 3-way parity).
  Characters choose between hotel A and hotel B. Results so far:
  - Qwen3-32B with LoRA (Slack, Nov 14; same code and defaults as `regexintro`: 4000 steps, warmup
    1600, eval every 400 steps). At step 4000:

    | Level | Decision accuracy | Faithfulness | Shuffled baseline |
    |---|---|---|---|
    | L1 one literal | 1.00 | 0.76 | ≈ 0.49 |
    | L2 AND/OR of two | 1.00 | 0.74 | ≈ 0.50 |
    | L3 2-term DNF | 0.99 | 0.58 | ≈ 0.50 |
    | L4 parity of three | 0.50 (never learned) | 0.50 | ≈ 0.50 |

    Faithfulness rose only after the 1600-step warmup. Training L3 and L4 longer didn't change this,
    and eventually broke the model's English.
  - Small GPT-2 trained from scratch, "pretrained" on choices plus reports for one group of
    characters, then given choices only for a new group: final faithfulness was L1 0.74, L2 0.72,
    L3 0.71, L4 0.58. Decision accuracy was 1.00 for L1–L3 and 0.95 for L4.
  - More reports in pretraining raised faithfulness but not decision accuracy.
  - A linear probe found the preference stored in the character embedding even with zero reports.
- **This variant (`regexintro/`).** The hidden rule is a regular grammar, and the model judges one
  string at a time. Regexes give a cleaner, multi-dimensional difficulty axis and an exactly
  checkable meta-answer.

## 2. Design

**Task.** Each character accepts the strings over {a, b, c} (length 1–8, 9,840 strings) that fully
match a hidden regex.

- **Decision task (trained).** "Imagine you are {name}. You accept some strings of the letters a,
  b and c and reject the others. Would you accept this string? String: a b c a". Answer Yes or No.
  Letters are separated by spaces so each is one token.
- **Two framings.** Every decision is also asked as "Would you reject this string?". Training mixes
  both framings. The model's P(accept) is the mean of P(Yes | accept?) and P(No | reject?), which
  removes a constant bias toward "Yes". This replaces Keshav's trick of showing each pair in both
  orders. The model accepts a string when P(accept) > 0.5.
- **Self-report task (never trained).** The model writes its rule as a regex, using only a, b, c
  and `. | ( ) [ ] * + ? {m,n}`. It gets 24 reports per character, each in a separate context with a
  different example string, sampled at temperature 0.7.

**Difficulty levels.** X, Y, Z is a random permutation of a, b, c for each character. 100
characters per run.

| Level | Kind | Templates |
|---|---|---|
| L1 | one constraint | `X.*`, `.*X`, `.*X.*`, `[YZ]+` |
| L2 | two constraints | `X.*Y`, `.*XY.*`, `X[XZ]*`, `.*X.*Y.*` |
| L3 | repeated blocks | `(X.\|Y)+`, `(.X\|Y)+`, `X+[YZ]+`, `[YZ]+X+` |
| L4 | counting / parity (analog of Keshav's XOR) | #X even, #X odd, #X ≡ 0 mod 3, #X ≡ 1 mod 3 |

No two templates define the same language under any letter permutation (this is tested). Each
template has at least 150 accepted and 150 rejected strings.

**Data per character.**
- 64 held-out eval strings: 32 accepted and 32 rejected. Sampling is length-stratified: pick a
  length uniformly among lengths where that label exists, then a uniform string of that length.
- The training stream draws a label 50/50, then a length-stratified string from the remaining
  9,776, then a random framing.

**Metrics** (`regexintro/metrics.py`, per character, then averaged).

| Metric | Meaning |
|---|---|
| `decision_acc` | model decision vs. hidden grammar on the eval strings |
| `model_accept_rate` | share of eval strings the model accepts; the target share is 0.5 |
| `faithfulness` (main) | share of eval strings where the reported regex accepts or rejects the same way the model did, averaged over parsed reports. Equivalent regexes get full credit. There are no ties, so chance ≈ 0.5 |
| `faithfulness_balanced` | mean agreement on model-accepted and on model-rejected strings, so `.*` can't win when the model accepts most strings |
| `report_target_agreement` | the reported regex vs. the hidden grammar |
| `exact_match` | the report defines exactly the hidden language (all strings up to length 8) |
| `parse_rate` | share of reports that parse as a valid regex. Unparsed reports are excluded from faithfulness, not scored 0 |
| `error_faithfulness` | faithfulness on strings where the model disagrees with the target. Above 0.5 means reports track the model's behaviour, not the training label. This is the cleanest evidence |

**Controls.** These are the same as Keshav's.
- **Shuffled baseline.** A character's reports are scored against other characters' behaviour at
  the same level, excluding characters with the identical grammar (L1 has only 12 distinct
  languages).
- **Base model.** Step 0, before training.
- **Error strings.** See `error_faithfulness` above.
- **Observer baseline.** The base model sees 16 of the trained model's accept/reject decisions for
  a character and writes the regex. If the observer does as well as self-report, the report needn't
  come from introspection.

**Training.** These are boolintro's defaults, unchanged.
- Qwen3-32B in bf16 with rank-8 LoRA (alpha 8) on all linear layers.
- 4000 steps, lr 1e-4 with 1600 warmup steps, batch 32 (micro-batch 16).
- A checkpoint every 200 steps; evaluation runs on every 400th.
- Loss is applied only to the Yes/No answer token.

## 3. Code

`regexintro/` reuses `boolintro.lora`, `boolintro.modeling`, checkpoint selection and the plotting
in `boolintro.analyze`. Never edit `boolintro/`; Keshav's runs depend on it.

| File | Role |
|---|---|
| `grammars.py` | string universe, level templates, random grammars, language vectors, lenient report parser (strips `^`/`$`, `/.../`, code fences, labels like "Regex:", and whitespace; lowercases) |
| `data.py` | spec (characters, grammars, eval strings), train pools, training stream, report and observer strings |
| `prompts.py` | decision, report and observer prompts (same system prompt as boolintro) |
| `metrics.py` | per-character metrics |
| `train.py` | LoRA training on decisions; writes `spec.json`, `train_config.json`, checkpoints |
| `evaluate.py` | decisions, self-reports and controls per checkpoint → `eval/step_XXXXXX.json` and `_reports.jsonl` (raw reports). Resumable |
| `rescore.py` | recompute metrics from saved outputs with no GPU, e.g. after a parser change |
| `analyze.py` | `report.md`, `summary.csv`, `final_contrasts.csv`, `trajectories.png`, `faithfulness_dist.png`, `self_vs_observer.png` |
| `preview.py` | print grammars and prompts; no GPU |
| `tests/test_regex.py` | 36 tests |
| `cluster/regex/` | `03_smoke.sbatch`, `10_train.sbatch`, `20_eval.sbatch`, `30_analyze.sbatch`, `submit_all.sh` |

Runs land in `$RUNS_ROOT/${REGEX_TAG}_L<level>`. The default tag is `regex-qwen3-32b`. It is a
separate variable from Keshav's `TAG`, so the two sets of runs never collide.

**Already verified (off-cluster).**
- All 74 tests pass (boolintro and regexintro).
- A full train → evaluate (base + 2 checkpoints + observer) → analyze run worked on a tiny,
  randomly initialized Qwen3-architecture model with a locally built tokenizer.

**Not yet verified.** The real Qwen3 tokenizer and model have not been tried. The code asserts
that "Yes" and "No" are single tokens. Whether Qwen3's reports parse well is unknown. The smoke
test checks both.

## 4. Run book (Torch)

Rules for whoever runs this (human or agent):
- Show the exact `sbatch` or `submit_all.sh` command and wait for approval before submitting.
- Don't edit `boolintro/` or Keshav's `cluster/*.sbatch`, and don't push to his GitHub repo.
- On any failure, show the log tail and propose a fix before changing code.

### Step 0 — Environment

```bash
cd /scratch/$USER/intros            # the repo, with regexintro/ extracted into its root
source cluster/env.sh
my_slurm_accounts                    # if torch_pr_1363_cds isn't listed: export TORCH_ACCOUNT=<yours>
ls -lh "$OVERLAY"                    # default: /scratch/$USER/section33_overlay-15GB-500K.ext3
mkdir -p logs
```

`env.sh` defaults come from Keshav's setup:
- If the account differs, set `TORCH_ACCOUNT`.
- If the overlay file is missing, build one by following the header of `cluster/01_setup_overlay.sh`.
  This uses an interactive `srun`, then `apptainer --fakeroot`. Then set
  `export OVERLAY=/scratch/$USER/intros_overlay-15GB-500K.ext3` in `env.sh`.

### Step 1 — Tests and preview (login node, inside the container)

```bash
apptainer exec --overlay "$OVERLAY:ro" "$SIF" /bin/bash -c \
  "source /ext3/env.sh && cd $REPO && python -m pytest -q tests && python -m regexintro.preview --n 8"
```

Expect `74 passed`.

### Step 2 — Prefetch models (login node; compute nodes have no internet)

```bash
bash $REPO/cluster/02_prefetch.sh    # Qwen3-32B (~65 GB) and Qwen3-0.6B into $HF_HOME
```

### Step 3 — Smoke test (Qwen3-0.6B, about 10 minutes)

```bash
sbatch --account="$TORCH_ACCOUNT" --chdir="$REPO" cluster/regex/03_smoke.sbatch
# then: tail -f logs/rx-smoke-<jobid>.out
```

Check in the log:
- tests pass;
- training loss falls;
- three eval lines (step 0, 30, 60) with decision_acc, faithfulness, parse and accept_rate;
- no "must be single tokens" error;
- the raw reports printed at the end.

A 0.6B model after 60 steps is not expected to do well. This only checks the plumbing.

**If the parse rate is low (under about 0.5).**
1. Look at raw reports: `$RUNS_ROOT/regex_smoke_<jobid>/L1/eval/step_000060_reports.jsonl`
   (the `reports` field).
2. Make `grammars.parse_report` / `normalise` more lenient for the formats that appear, and add a
   test case for each.
3. Recompute without a GPU:
   `python -m regexintro.rescore --run-dir $RUNS_ROOT/regex_smoke_<jobid>/L1`
4. If a large share of reports are prose ("strings that start with a"), raise it: a prompt change
   needs approval, because it changes the experiment.

### Step 4 — Main experiment (after the smoke test is approved)

```bash
bash $REPO/cluster/regex/submit_all.sh
```

This chains three jobs:
1. Train: array 1–4, one H200 per level, about 4–6 hours each.
2. Eval: array 1–4. Task i starts when training task i finishes. It covers base + every 400th step
   + final, with the observer at the final step.
3. Analyze: all four levels together.

Watch with `squeue --me` and `tail -f logs/rx-train-<jobid>_<level>.out`. Training and eval are
both resumable: resubmitting the same command continues from the last checkpoint or step.

Results: `$RUNS_ROOT/regex-qwen3-32b_analysis/report.md` and the figures.

### Optional variants (each needs approval)

```bash
MODEL=Qwen/Qwen3-8B REGEX_TAG=regex-qwen3-8b bash $REPO/cluster/regex/submit_all.sh       # scale
LEVELS=1,2,3,4 ARRAY= REGEX_TAG=regex-qwen3-32b-mixed bash $REPO/cluster/regex/submit_all.sh   # one mixed adapter
```

The 8B run uses the same H200 nodes; no script change is needed. For longer training of hard
levels, set `TRAIN_ARGS="--steps 12000"` with `ARRAY=3-4`.
Prefetch any extra model first:
`EXTRA_MODELS="Qwen/Qwen3-8B" bash $REPO/cluster/02_prefetch.sh`.

## 5. Reading the results

For each level, report from `report.md`:
- decision accuracy and accept rate (it should be about 0.5; far from that means one-sided behaviour);
- faithfulness and balanced faithfulness;
- faithfulness minus shuffled, and minus base (with bootstrap CIs);
- error-string faithfulness minus 0.5, with its n;
- exact match;
- observer faithfulness, and self minus observer.

Then describe `trajectories.png` (metrics over training steps) and `self_vs_observer.png`.

| Pattern | Reading |
|---|---|
| High decision acc, faithfulness ≈ shuffled ≈ 0.5 | the model learned the rule but can't report it (close to Keshav's Qwen L3: 0.99 vs 0.58) |
| Faithfulness well above shuffled and base, and error-string faithfulness > 0.5 | evidence that reports track the model's own behaviour |
| Self ≈ observer | the report could have come from watching behaviour; no privileged access shown |
| Faithfulness high but balanced ≈ 0.5 | reports are mostly `.*`-like and match a model that accepts most strings |
| Faithfulness drops from L1 to L4 | difficulty limits verbalization, as with boolean parity |

If decision accuracy stays near 0.5 (as for his L4), the model never learned the rule and faithfulness
at that level means nothing. Compare each level with Keshav's Qwen3-32B table in section 1 (the
direct counterpart of this run). Keep in mind that his small-GPT-2
setup differs (it was trained from scratch with reports for some characters).

## 6. Open items (not built yet)

- **Conflict test** (boolintro `40_conflict.sbatch`: forged records of past choices in context).
  Not ported to regex.
- **Small GPT-2 from scratch** (Keshav's Nov 16–18 runs: class-1 / class-2 split, sweep over the
  number of reports, linear probe). His code isn't in this repo; ask him for it or port it to regex.
- **Shashwat's question.** Can base Qwen3 follow each rule when it is stated explicitly in the
  prompt? This is an easy extra check: put the regex in the decision prompt and score the base
  model's decision accuracy.
