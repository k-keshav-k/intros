# Regex introspection experiment (NYU Torch)

## Project
- Ayush Kokande (NetID ak13124), NYU independent study on LLM introspection.
- This repo is Keshav Krishna's `github.com/k-keshav-k/intros`, master at `9345431` ("gpt 2 exp"). `boolintro/` is his. `README.md` is his version, unchanged.
- `regexintro/`, `tests/test_regex.py`, `cluster/regex/` and `REGEX_EXPERIMENT.md` come from `regexintro.tar.gz`, extracted over the repo. On the Mac they're on branch `regex-variant`, pushed to the user's fork `github.com/ayushkokande/intros` (remote `origin`, which the branch tracks). Remote `upstream` is Keshav's repo: never push there, no PR to it unless asked.
- `REGEX_EXPERIMENT.md` is the design and run book. Section 4 lists the Torch steps; section 5 says how to report results. Read it before running anything.

## Ground rules
- Never edit `boolintro/` or Keshav's `cluster/*.sbatch`. Never push to his GitHub repo.
- Show the exact `sbatch` / `submit_all.sh` command and wait for an explicit OK before submitting.
- On any failure, show the log tail and the proposed fix before changing code.
- After the smoke test, show its results and ~10 raw reports from `step_000060_reports.jsonl`, then stop and wait for approval before the main experiment.
- Results: summarize each level as in section 5. Compare first with Keshav's boolean Qwen3-32B run, the direct counterpart (same code and defaults, step 4000). Its shuffled baseline is about 0.49–0.50:

  | Level | Decision acc | Faithfulness |
  |---|---|---|
  | L1 | 1.00 | 0.76 |
  | L2 | 1.00 | 0.74 |
  | L3 | 0.99 | 0.58 |
  | L4 | 0.50 (never learned) | 0.50 |

  If decision accuracy stays near 0.5, faithfulness at that level means nothing. His small GPT-2 run (trained from scratch, with reports for some characters, so a different setup) got faithfulness L1 0.74, L2 0.72, L3 0.71, L4 0.58.
- Run compute only through sbatch/srun, except the run book's login-node steps (tests and preview in the container, model prefetch). Keep large outputs in scratch.

## Where things are
| What | Where |
|---|---|
| Mac repo | `/Users/kokande/Desktop/intros`, venv `.venv/` (Python 3.12). 134 tests pass. |
| Torch repo | `/scratch/ak13124/intros` |
| Runs | `$RUNS_ROOT` = `/scratch/ak13124/intros_runs`. Smoke: `regex_smoke_<jobid>/L1`. Main: `regex-qwen3-32b_L{1,2,3,4}`. Combined: `regex-qwen3-32b_analysis/` |
| Overlay | `$OVERLAY`, default `/scratch/ak13124/section33_overlay-15GB-500K.ext3` |
| Slurm account | `torch_pr_1363_cds`, the "confidence" allocation and the `env.sh` default. Use it for regexintro and for the probe job. A test GPU job ran on ga004 (A100). |
| SSH | `ssh torch` goes to `login.torch.hpc.nyu.edu` (needs NYU VPN; Microsoft device login then Duo; SSH keys not supported). The DTN `dtn.torch.hpc.nyu.edu` is for rsync. Login nodes present different host keys. |
| Local results | `results/regex-qwen3-32b/` (rsync the analysis dir, not the adapters) |
| Overlay build | Built at `/scratch/ak13124/intros_overlay-15GB-500K.ext3`: torch 2.15.0+cu132 (CUDA 13.2), transformers 5.17.0. The A100 driver supports up to CUDA 13.3; check `nvidia-smi` on H200 nodes in the smoke log. The Torch clone's `cluster/env.sh` defaults `OVERLAY` to this file (local edit, not pushed). |
| Checksums | `regexintro.tar.gz` sha256 `34430ab2b9c79d3919b552f6510e12e8b0e76877b1ce308be71b9d6133e0436b` (contains no README). `REGEX_EXPERIMENT.md` md5 `1daa3d345d6969cff82714b22792d238`. Combined md5 of `regexintro/*.py` and `cluster/regex/*` (15 files, C-sorted) is `cb489f3bc18ef23d6c5b87a3f5209c86` |

## Status (2026-11-22 evening)
- Torch repo updated to `9345431` plus the regex files. Run book Steps 0–3 done: Step 1 passed 87 tests in the container; models are prefetched. The Qwen3-0.6B smoke test (job 24127029) and the Qwen3-32B smoke test on H200 (24127373) both passed. 32B reports parse at 0.85–0.95.
- **Main experiment submitted** (Step 4): train `24127370_[1-4]` (about 1.8 h per level at 1.55 s/step) → eval `24127371_[1-4]` → analyze `24127372`.
- **Ports of Keshav's other experiments** (new files only; existing `regexintro/` modules untouched while jobs import them):
  - conflict test: `regexintro/conflict.py`, `conflict_analyze.py`, `cluster/regex/40`, `41`
  - explicit-rule check: `regexintro/explicit.py`, `cluster/regex/60`
  - report diagnosis with the shuffled error-string baseline and a per-letter view (regex version of Keshav's attribute diagnosis: sign_ok, rel/irr_split by letter-swap symmetry, valence correlation with shuffled baseline, assoc_ceiling, presence_r2): `regexintro/diagnose.py`, `cluster/regex/35`
  - template identification helper: `regexintro/families.py`
  - from-scratch GPT-2 sandbox: `regexintro/tiny/`, `cluster/regex/50`, `51`
  - tiny extras (`2821288`): attribute world, `maj`/`mux` families, `summary_plots.py`, `cluster/regex/52`, `53`. Default behaviour byte-identical to before (golden-digest tests).
  - tests: `tests/test_regex_extras.py`, `tests/test_regex_tiny.py`, `tests/test_regex_tiny_worlds.py`; 129 pass on the Mac
  - On Torch via `git restore --source=fork/regex-variant --worktree` (remote `fork` = the user's fork); 115 tests pass in the container.
  - Submitted 2026-11-22 ~18:30: explicit `24130435` (starts 10 min after L4 train starts), tiny `24130436_[0-2]`, tiny sweep `24130438_[0-5]`, diagnose `24130439` (after all eval), conflict `24130440_[1-4]` (aftercorr eval), conflict analyze `24130441`. Tiny B and C finished (results in `results/regex-tiny/`, weights left on scratch).
  - Tiny extras run from a separate worktree `/scratch/ak13124/intros_tiny2` at `2821288` (its `env.sh` has Keshav's OVERLAY default, so pass `OVERLAY` and `REPO`): worlds `24136099_[0-5]`, summary `24136100` (after worlds and diagnose).
- **Overnight authority (user, 2026-11-22 ~19:45; user reviews in the morning and does nothing before):** fix and resubmit failed jobs; start longer L3/L4 training if needed; commit and push to the fork (`origin`) as needed. Still never push to `upstream`. Deliverable: `results/SUMMARY.md` plus a short "read this first" message. Watcher: background `scratchpad/watch_jobs<N>.sh` (session scratchpad); Mac kept awake with `caffeinate`.
  - Longer training rule (per level, L3/L4): continue in place to 12000 steps (`ARRAY=<n> TRAIN_ARGS="--steps 12000" EVAL_ARGS="--every 800"`) if held-out decision accuracy at step 4000 is < 0.90 and still rising (≥ +0.02 over the last 1200 steps), or if faithfulness is still rising ≥ +0.03 over the last 1200 steps. Before continuing a level in place, make sure its conflict job has resolved its step (it uses the final step) or pin `--step 4000`.
  - **L3 extended** (faithfulness +0.033 over steps 2800→4000): train `24146182_3` (`--steps 12000`, in place) → eval `24146183_3` (`--every 800`; its per-run analysis overwrites `regex-qwen3-32b_L3/analysis`). Submitted with `sbatch` directly, not `submit_all.sh`, whose analyze step would overwrite the official combined `regex-qwen3-32b_analysis`. Step-4000 numbers stay the main comparison; local copies of the step-4000 per-level analyses are in `results/regex-qwen3-32b/step4000/`. Anything that defaults to "final step" (diagnose, conflict, analyze) picks step 12000 for L3 once that eval lands: pass `--ref-step 4000` / `--step 4000` when rerunning.
- **Prefill rerun** (user OK, 2026-11-22 ~22:00): `--prefill` in `regexintro.evaluate`/`conflict` (`e3e32de`, off by default) starts each report with `prompts.REPORT_PREFILL` ("Regular expression:"), because late-training reports collapse into "Yes/No" answers. Runs from worktree `/scratch/ak13124/intros_prefill` at `e3e32de` via `cluster/regex/70_prefill.sh` (pass `OVERLAY` and `REPO`); view run dirs `regex-qwen3-32b-prefill_L{1..4}` symlink the original checkpoints ≤ 4000 (originals only read). Jobs: eval `24148043_[1-4]` (`--every 400`) → analyze `24148044`, diagnose `24148050`; conflict `24148080_[1-4]` (`--step 4000`) → conflict analyze `24148090`. Outputs `regex-qwen3-32b-prefill_{analysis,diagnosis,conflict_analysis}`. Report next to the original (no prefill = Keshav's protocol, still the official comparison).
  - Results (2026-11-23): prefill fixes the collapse (parse 0.92–0.98); faithfulness L1–L4 0.677/0.613/0.604/0.566; L4 self − observer +0.048 [0.036, 0.060]. L3 12000-step run: no real gain (0.58–0.61), parse ≤ 4% after step 7200. Prefill conflict L1–L3 done (L3 resists forged records, 9 chars); L4 (`24148080_4`) and analysis `24148090` still queued at 08:45. Write-up: `results/SUMMARY.md`.
  - L3/L4 prefill conflict at `--min-faith 0.6` (user OK 2026-11-23 09:05; L4 had 0 characters at 0.7, L3 only 9): view dirs `regex-qwen3-32b-prefill06_L{3,4}` (symlink step_004000 and the prefill eval), conflict `24196736_[3-4]` → analysis `24196737` (`CONFLICT_LEVELS="3 4"`), output `regex-qwen3-32b-prefill06_conflict_analysis`. All completed by 10:35, no failures. L3 (43 chars) report resistance +0.25 [0.11, 0.39], concentrated in the 9 chars ≥ 0.7 (+0.62); L4 (28) labelled forged records steer decisions more than base (−0.40). Local copy `results/regex-qwen3-32b/prefill06/`; write-up in SUMMARY 4b.
- **L4 second seed** (user OK 2026-11-23 12:28, "do it"; checks L4 self > observer, +0.048 faithfulness and +0.36 valence on seed 2025): tag `regex-qwen3-32b-s2`, L4 only, `TRAIN_ARGS="--seed 2026"` (new characters, LoRA init and data order), `EVAL_ARGS="--prefill --every 400"`, from worktree `intros_prefill`. Jobs: train `24226420_4` → eval `24226421_4` → analyze `24226426`, diagnose `24226427`. Outputs `regex-qwen3-32b-s2_{L4,analysis,diagnosis}`. Compare with `regex-qwen3-32b-prefill_L4` (seed 2025, prefill). No conflict test or `--observer-k 64` unless asked.
  - Results (all completed 15:37): decision acc 1.000, faithfulness 0.615 (shuffled 0.494), observer 0.515, self − observer +0.101 [0.088, 0.114] (seed 2025: +0.048). Letter valence self − observer, equal-count over 200 draws: +0.79 [0.65, 0.93] (seed 2025: +0.32). Replicates. The observer is near chance at L4 on both seeds, so the open check is `--observer-k 64`. Local copy `results/regex-qwen3-32b/seed2/` (when rsyncing a run dir, exclude both `checkpoints/` and `resume/`, since `resume/` holds the adapter); write-up in SUMMARY 1d.
- SSH: `ControlPersist 8h` means the session expires 8 h after the user's interactive login; after that `ssh -o BatchMode=yes torch` fails until the user runs `ssh torch` in a real terminal (the `!` prompt can't do the Microsoft/Duo login).
- Not ported: Word2Vec (no code in the repo).
- Parser side check (local, copies only, `grammars.py` unchanged): allowing `$` anywhere and reading `^$` as reject-all changes L1/L2 faithfulness by ≤ 0.008. Low L2–L4 parse rates late in training come from the adapter answering "Yes/No …" to the report prompt, not from the parser.
- Reading results: error-string faithfulness and faith − base are inflated when the model is one-sided (low accept rate). Use balanced faithfulness and `diagnose`'s shuffled error baseline.

A separate task is pending: a probe job (`run.slurm`, conda env `probe_conf`, account above). The user still has to give the script and arguments.
