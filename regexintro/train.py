"""Train a multi-character LoRA adapter on the single-string decision task only.

The model never sees a grammar: it only sees Yes/No accept (or reject) decisions on
strings, labelled by each character's hidden regex.  Self-report is never trained.

    python -m regexintro.train --run-dir runs/regex_L1 --model Qwen/Qwen3-32B --levels 1

Resumable: rerunning the same command continues from the last saved step.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from boolintro import lora, modeling

from . import data, prompts


def parse_layers(s: str | None) -> set[int] | None:
    if not s:
        return None
    out: set[int] = set()
    for part in s.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return out


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-32B")
    p.add_argument("--levels", default="1", help="comma list, e.g. 1 or 1,2,3,4 (mixed)")
    p.add_argument("--n-characters", type=int, default=100)
    p.add_argument("--n-eval", type=int, default=64, help="eval strings per character (half accepted)")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=32, help="effective batch size")
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=1600)
    p.add_argument("--scheduler", choices=["constant", "cosine"], default="constant")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--layers", default=None, help="LoRA only on these layers, e.g. 0-19")
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
    p.add_argument("--no-grad-ckpt", action="store_true")
    p.add_argument("--train-eos", action="store_true",
                   help="also train the end-of-turn token after the letter (standard chat SFT); "
                        "default trains only the Yes/No token, as the eval reads one token")
    return p.parse_args(argv)


def lr_lambda(args):
    def f(step: int) -> float:
        if step < args.warmup:
            return (step + 1) / args.warmup
        if args.scheduler == "constant":
            return 1.0
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


def main(argv=None):
    args = get_args(argv)
    run = args.run_dir
    run.mkdir(parents=True, exist_ok=True)
    levels = tuple(int(x) for x in args.levels.split(","))

    spec_path = run / "spec.json"
    if spec_path.exists():
        spec = data.load_spec(spec_path)
        if tuple(spec["levels"]) != levels or spec["seed"] != args.seed:
            raise SystemExit(f"{spec_path} exists with levels={spec['levels']} seed={spec['seed']}; "
                             "use a new --run-dir")
    else:
        spec = data.build_spec(args.n_characters, levels, args.seed, args.n_eval)
        data.save_spec(spec, spec_path)
    has_ckpts = any((run / "checkpoints").glob("step_*/adapter.safetensors"))
    if has_ckpts and not (run / "resume" / "state.pt").exists():
        raise SystemExit(f"{run} has checkpoints but no resume/state.pt; refusing to restart from step 0 "
                         "and overwrite them")
    cfg_path = run / "train_config.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps({k: str(v) if isinstance(v, Path) else v
                                        for k, v in vars(args).items()}, indent=1))
    else:
        # Continuing an existing run: the optimizer and scheduler are restored from the saved
        # state, so changed hyperparameters would be silently ignored.  Only --steps may grow.
        saved = json.loads(cfg_path.read_text())
        fixed = ["model", "lr", "warmup", "scheduler", "weight_decay", "rank", "alpha", "layers",
                 "batch_size", "micro_batch", "quant", "train_eos"]
        diff = {k: (saved.get(k), getattr(args, k)) for k in fixed
                if k in saved and saved.get(k) != getattr(args, k)}
        if diff:
            raise SystemExit(f"{run} was trained with different settings {diff} (saved, given); "
                             "use a new --run-dir for a different configuration")
        if args.scheduler == "cosine" and args.steps != saved["steps"]:
            raise SystemExit("extending a cosine-schedule run changes its schedule; use a new --run-dir")
        if args.steps > saved["steps"]:
            saved.setdefault("extended_from", []).append(saved["steps"])
            saved["steps"] = args.steps
            cfg_path.write_text(json.dumps(saved, indent=1))
            print(f"extending training from {saved['extended_from'][-1]} to {args.steps} steps", flush=True)

    if args.batch_size % args.micro_batch:
        raise SystemExit("--batch-size must be a multiple of --micro-batch")
    accum = args.batch_size // args.micro_batch

    print(f"loading {args.model} (quant={args.quant})", flush=True)
    model, tok = modeling.load(args.model, args.quant)
    torch.manual_seed(args.seed)
    lcfg = lora.inject(model, args.rank, args.alpha, layers=parse_layers(args.layers))
    lcfg["base_model"] = args.model
    params = lora.lora_parameters(model)
    print(f"LoRA: {lcfg['n_wrapped']} modules, {sum(p.numel() for p in params) / 1e6:.1f}M params",
          flush=True)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.99), eps=1e-8,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(args))

    start = 0
    resume = run / "resume"
    if (resume / "state.pt").exists():
        st = torch.load(resume / "state.pt", map_location="cpu", weights_only=False)
        lora.load_into(model, resume)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        start = st["step"]
        print(f"resumed at step {start}", flush=True)
    if start >= args.steps:
        print("training already complete")
        return

    ia, ib = modeling.token_ids(tok, ("Yes", "No"))
    eot = modeling.end_of_turn_id(tok)
    stream = data.training_stream(spec, args.steps * args.batch_size, seed=args.seed + 1)
    chars = spec["characters"]
    dev = modeling.input_device(model)
    log = open(run / "train_log.jsonl", "a")
    t0 = time.time()
    run_loss, run_acc, run_n = 0.0, 0.0, 0

    for step in range(start, args.steps):
        exs = stream[step * args.batch_size:(step + 1) * args.batch_size]
        for m in range(accum):
            mb = exs[m * args.micro_batch:(m + 1) * args.micro_batch]
            texts = [modeling.render(tok, prompts.decision_user(chars[ci], s, fr), prompts.SYSTEM)
                     for ci, s, fr, _ in mb]
            labels = torch.tensor([ia if lab == "Yes" else ib for *_, lab in mb], device=dev)
            if args.train_eos:
                texts = [t + lab for t, (*_, lab) in zip(texts, mb)]
                both = modeling.last_logits(model, modeling.encode(tok, texts, dev), n=2).float()
                logits = both[:, 0]
                loss = 0.5 * (F.cross_entropy(logits, labels)
                              + F.cross_entropy(both[:, 1], torch.full_like(labels, eot)))
            else:
                logits = modeling.last_logits(model, modeling.encode(tok, texts, dev)).float()
                loss = F.cross_entropy(logits, labels)
            (loss / accum).backward()
            with torch.no_grad():
                pred = torch.where(logits[:, ia] > logits[:, ib], ia, ib)
                run_acc += (pred == labels).float().sum().item()
            run_loss += loss.item() * len(mb)
            run_n += len(mb)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

        done = step + 1
        if done % args.log_every == 0 or done == args.steps:
            rec = {"step": done, "loss": run_loss / run_n, "train_acc": run_acc / run_n,
                   "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            run_loss, run_acc, run_n = 0.0, 0.0, 0
        if done % args.save_every == 0 or done == args.steps:
            lora.save(model, run / "checkpoints" / f"step_{done:06d}")
            lora.save(model, resume)
            torch.save({"step": done, "optimizer": opt.state_dict(), "scheduler": sched.state_dict()},
                       resume / "state.pt.tmp")
            (resume / "state.pt.tmp").replace(resume / "state.pt")
            print(f"saved step {done}", flush=True)
    log.close()
    (run / "TRAINING_DONE").write_text(str(args.steps))


if __name__ == "__main__":
    main()
