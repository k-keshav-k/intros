"""Pretrain and finetune a small, randomly initialised GPT-2 on regex decisions and reports.

Port of boolintro.tiny.train (same stages, defaults and flags; the model, save/load, loss and
LR schedule are boolintro.tiny.model).

Stage 1 (pretrain): decisions + reports for "pre_report" characters, decisions only for
"pre_choice" characters.  Builds the spec and the model.  --report-format formula (default:
the full regex) | attribute (one letter the character cares about) | both; --families from
L1,L2,L3,L4,maj,mux (default L1-L4).

    python -m regexintro.tiny.train pretrain --run-dir runs/regex_tiny/formula_s0
    python -m regexintro.tiny.train pretrain --run-dir runs/regex_tiny/attribute-all_s0 \
        --report-format attribute --families L1,L2,L3,L4,maj,mux

Stage 2 (finetune): decisions only for the "new" characters.  Their character embeddings are
re-drawn at random first.  --train embed (default) updates only those rows, so the report
machinery learned in stage 1 stays fixed; --train all updates the whole model (add
--replay 0.5 to keep half of each batch on stage-1 data).

    python -m regexintro.tiny.train finetune --run-dir runs/regex_tiny/formula_s0 --train embed

Both stages evaluate every group at step 0 and every --eval-every steps.  Not resumable:
rerunning a finished stage is a no-op, an unfinished one starts over.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from boolintro.tiny import model as M

from . import data as D
from . import evaluate as E

PRE_DEFAULTS = dict(steps=30000, lr=5e-4, warmup=1000, weight_decay=0.01, eval_every=2000)
FT_DEFAULTS = dict(steps=5000, lr=1e-3, warmup=100, weight_decay=0.0, eval_every=250)


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)
    for name, dflt in (("pretrain", PRE_DEFAULTS), ("finetune", FT_DEFAULTS)):
        s = sub.add_parser(name)
        s.add_argument("--run-dir", type=Path, required=True)
        s.add_argument("--device", default="auto")
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--batch-size", type=int, default=256)
        s.add_argument("--steps", type=int, default=dflt["steps"])
        s.add_argument("--lr", type=float, default=dflt["lr"])
        s.add_argument("--warmup", type=int, default=dflt["warmup"])
        s.add_argument("--weight-decay", type=float, default=dflt["weight_decay"])
        s.add_argument("--eval-every", type=int, default=dflt["eval_every"])
        s.add_argument("--log-every", type=int, default=50)
        s.add_argument("--n-samples", type=int, default=8, help="sampled reports per character (plus greedy)")
        s.add_argument("--temperature", type=float, default=1.0)
        s.add_argument("--n-peers", type=int, default=3, help="characters per shuffled baseline")
        if name == "pretrain":
            s.add_argument("--families", default="L1,L2,L3,L4",
                           help="rule families (dealt out evenly in each group): " + ",".join(D.FAMILIES)
                                + ".  boolintro.tiny's default also has maj,mux; ours stays L1-L4")
            s.add_argument("--report-format", choices=list(D.REPORT_KINDS), default="formula",
                           help="pretraining reports: the full regex, one weighted letter, or both")
            s.add_argument("--n-with-reports", type=int, default=None,
                           help="sweep mode: all n-pre-report + n-pre-choice pretraining characters are "
                                "built as one group and only the first N keep their reports")
            s.add_argument("--report-families", default=None,
                           help="only these levels get reports in pretraining, e.g. L1,L2 (or L1+L2)")
            s.add_argument("--n-pre-report", type=int, default=1000)
            s.add_argument("--n-pre-choice", type=int, default=200)
            s.add_argument("--n-new", type=int, default=200)
            s.add_argument("--n-eval", type=int, default=64, help="held-out eval strings per character")
            s.add_argument("--holdout-rule-frac", type=float, default=0.0,
                           help="reserve this fraction of languages for new characters only")
            s.add_argument("--report-frac", type=float, default=0.25, help="share of each batch that is reports")
            s.add_argument("--n-layer", type=int, default=6)
            s.add_argument("--n-embd", type=int, default=256)
            s.add_argument("--n-head", type=int, default=8)
            s.add_argument("--dropout", type=float, default=0.0)
        else:
            s.add_argument("--name", default=None, help="output subdir (default ft_<train>_s<seed>)")
            s.add_argument("--train", choices=["embed", "all"], default="embed")
            s.add_argument("--replay", type=float, default=0.0,
                           help="share of each batch drawn from stage-1 data (decisions + reports)")
            s.add_argument("--replay-report-frac", type=float, default=0.25)
            s.add_argument("--no-reinit", action="store_true",
                           help="keep the new characters' (never trained) initial embeddings")
    args = p.parse_args(argv)
    if args.stage == "finetune" and args.name is None:
        args.name = f"ft_{args.train}" + (f"_replay{args.replay:g}" if args.replay else "") + f"_s{args.seed}"
    return args


def _fams(s: str) -> tuple[str, ...]:
    return tuple(f"L{x}" if x.isdigit() else x for x in s.replace("+", ",").split(","))


def run_eval(model, spec, dev, args, step, f_sum, f_chr):
    t = time.time()
    per, summ = E.evaluate(model, spec, D.GROUPS, dev, args.n_samples, args.temperature,
                           seed=args.seed * 100003 + step, n_peers=args.n_peers)
    for rows, f in ((summ, f_sum), (per, f_chr)):
        for r in rows:
            r.update(step=step, stage=args.stage)
            f.write(json.dumps(r) + "\n")
        f.flush()
    p = D.KIND_PREFIX[D.report_kinds(spec)[0]]
    brief = "  ".join(f"{r['group']}/{r['level']}: dec={r['decision_acc']:.2f} "
                      f"faith={r[p + 'faithfulness']:.2f} shuf={r[p + 'shuffled_faithfulness']:.2f} "
                      f"parse={r[p + 'parse_rate']:.2f}" for r in summ)
    print(f"[eval step {step} {time.time() - t:.0f}s] {brief}", flush=True)


def train_loop(model, params, next_batch, spec, dev, args, out: Path):
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, M.lr_lambda(args.warmup, args.steps))
    f_log = open(out / "train_log.jsonl", "w")
    f_sum = open(out / "eval.jsonl", "w")
    f_chr = open(out / "eval_chars.jsonl", "w")
    model.train()
    run_eval(model, spec, dev, args, 0, f_sum, f_chr)
    t0 = time.time()
    zero = {"loss": 0.0, "dec_correct": 0, "dec_n": 0, "rep_loss": 0.0, "rep_n": 0, "n": 0}
    acc = dict(zero)
    for step in range(args.steps):
        x, y, attn = next_batch()
        loss, logits = M.lm_loss(model, x, y, attn)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            dec = x[:, 2] == D.TOK["DEC"]
            if dec.any():
                pos = (y[dec] != -100).float().argmax(1)          # the answer position
                rows = torch.arange(int(dec.sum()), device=x.device)
                lg = logits[dec][rows, pos - 1][:, [D.YES, D.NO]]
                pred = torch.where(lg[:, 0] > lg[:, 1], D.YES, D.NO)
                acc["dec_correct"] += (pred == x[dec][rows, pos]).sum().item()
                acc["dec_n"] += int(dec.sum())
            if (~dec).any():
                rl = torch.nn.functional.cross_entropy(
                    logits[~dec, :-1].reshape(-1, logits.size(-1)).float(), y[~dec, 1:].reshape(-1),
                    ignore_index=-100)
                acc["rep_loss"] += rl.item()
                acc["rep_n"] += 1
        acc["loss"] += loss.item()
        acc["n"] += 1
        done = step + 1
        if done % args.log_every == 0 or done == args.steps:
            rec = {"step": done, "loss": acc["loss"] / acc["n"],
                   "train_dec_acc": acc["dec_correct"] / max(1, acc["dec_n"]),
                   "report_loss": acc["rep_loss"] / acc["rep_n"] if acc["rep_n"] else None,
                   "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
            f_log.write(json.dumps(rec) + "\n")
            f_log.flush()
            if done % (args.log_every * 10) == 0 or done == args.steps:
                print(json.dumps(rec), flush=True)
            acc = dict(zero)
        if done % args.eval_every == 0 or done == args.steps:
            run_eval(model, spec, dev, args, done, f_sum, f_chr)
    for f in (f_log, f_sum, f_chr):
        f.close()


def _write_config(out: Path, args) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                                 for k, v in vars(args).items()}, indent=1))


def pretrain(args):
    run = args.run_dir
    out = run / "pretrain"
    if (out / "DONE").exists():
        print(f"{out} already done")
        return
    spec_path = run / "spec.json"
    n_rep, n_cho = args.n_pre_report, args.n_pre_choice
    if args.n_with_reports is not None:
        n_rep, n_cho = n_rep + n_cho, 0
    spec = D.build_spec(n_rep, n_cho, args.n_new, _fams(args.families), args.seed, args.n_eval,
                        args.holdout_rule_frac,
                        _fams(args.report_families) if args.report_families else None,
                        args.n_with_reports, report_format=args.report_format)
    if spec_path.exists() and D.load_spec(spec_path) != spec:
        raise SystemExit(f"{spec_path} exists with different settings; use a new --run-dir")
    D.save_spec(spec, spec_path)
    _write_config(out, args)
    dev = M.device_of(args.device)
    model = M.make_model(D.vocab_size(spec), args.n_layer, args.n_embd, args.n_head, args.dropout,
                         seed=args.seed).to(dev)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params on {dev}, "
          f"vocab {D.vocab_size(spec)}", flush=True)
    sampler = D.Sampler(spec, D.ids_of(spec, ("pre_report", "pre_choice")),
                        D.ids_of(spec, ("pre_report",)), seed=args.seed + 1, kinds=D.report_kinds(spec))
    next_batch = lambda: D.collate(sampler.batch(args.batch_size, args.report_frac), dev)  # noqa: E731
    train_loop(model, list(model.parameters()), next_batch, spec, dev, args, out)
    M.save(model, out / "model")
    (out / "DONE").write_text(str(args.steps))


def finetune(args):
    run = args.run_dir
    out = run / args.name
    if (out / "DONE").exists():
        print(f"{out} already done")
        return
    if not (run / "pretrain" / "DONE").exists():
        raise SystemExit(f"run the pretrain stage for {run} first")
    spec = D.load_spec(run / "spec.json")
    _write_config(out, args)
    dev = M.device_of(args.device)
    model = M.load(run / "pretrain" / "model", dev)
    new_ids = D.ids_of(spec, ("new",))
    rows = torch.tensor([D.char_token(c) for c in new_ids], device=dev)
    wte = model.transformer.wte.weight
    if not args.no_reinit:  # fresh random embeddings for the new characters
        g = torch.Generator(device="cpu").manual_seed(args.seed + 7)
        fresh = torch.randn(len(new_ids), wte.shape[1], generator=g) * model.config.initializer_range
        with torch.no_grad():
            wte[rows] = fresh.to(dev, wte.dtype)
    if args.train == "embed":
        if args.weight_decay:
            raise SystemExit("--train embed needs --weight-decay 0 (AdamW would decay every embedding row)")
        for p in model.parameters():
            p.requires_grad_(False)
        wte.requires_grad_(True)
        mask = torch.zeros(wte.shape[0], 1, device=dev)
        mask[rows] = 1.0
        wte.register_hook(lambda grad: grad * mask)
        params = [wte]
    else:
        params = list(model.parameters())
    print(f"finetune {args.train}: {sum(p.numel() for p in params) / 1e6:.2f}M trainable "
          f"({len(new_ids)} new characters), replay={args.replay}", flush=True)

    new_s = D.Sampler(spec, new_ids, [], seed=args.seed + 11)
    rep_s = None
    if args.replay > 0:
        rep_s = D.Sampler(spec, D.ids_of(spec, ("pre_report", "pre_choice")),
                          D.ids_of(spec, ("pre_report",)), seed=args.seed + 13, kinds=D.report_kinds(spec))
    n_rep = int(round(args.batch_size * args.replay))

    def next_batch():
        parts = new_s.batch(args.batch_size - n_rep)
        if n_rep:
            parts += rep_s.batch(n_rep, args.replay_report_frac)
        return D.collate(parts, dev)

    train_loop(model, params, next_batch, spec, dev, args, out)
    M.save(model, out / "model")
    (out / "DONE").write_text(str(args.steps))


def main(argv=None):
    args = get_args(argv)
    (pretrain if args.stage == "pretrain" else finetune)(args)


if __name__ == "__main__":
    main()
