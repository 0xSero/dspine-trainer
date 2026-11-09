"""Train a DSpine drafter from captured hidden states."""
import argparse
import json
import math
import os
import random
import time

import torch
from safetensors.torch import load_file, save_file

from .common import draft_model, is_heldout, load_config, load_shard, online_model, shard_files, target_modules, token_ids


def read_weights(source):
    """A local .safetensors file, or a Hugging Face repo id holding model.safetensors."""
    if os.path.isfile(source):
        return load_file(source)
    from huggingface_hub import hf_hub_download
    return load_file(hf_hub_download(source, "model.safetensors"))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--config", default="configs/glm-5.3-flash.json")
    a.add_argument("--data", required=True, help="directory of capture shards")
    a.add_argument("--out", required=True)
    a.add_argument("--steps", type=int, default=3000)
    a.add_argument("--batch", type=int, default=4)
    a.add_argument("--anchors", type=int, default=64, help="block positions per step; SpecForge's default is 512")
    a.add_argument("--lr", type=float, default=6e-4)
    a.add_argument("--backbone-lr", type=float, default=None, help="learning rate for weights loaded by --warm-from or --resume")
    a.add_argument("--warm-from", default="", help="repo id or file with compatible weights. The result inherits that license.")
    a.add_argument("--resume", default="", help="continue from one of this trainer's own checkpoints")
    a.add_argument("--save-every", type=int, default=200)
    a.add_argument("--holdout-every", type=int, default=20, help="keep 1 in N shards out of training for evaluate.py; 0 keeps none")
    a.add_argument("--seed", type=int, default=1)
    a.add_argument("--device", default="cuda")
    args = a.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dev = args.device
    cfg = load_config(args.config)
    with open(f"{args.out}/config.json", "w") as f:
        json.dump(cfg["draft"], f, indent=1)
    assistant, ends = token_ids(cfg)
    draft = draft_model(cfg, dev)
    state = draft.state_dict()

    loaded = set()
    if args.warm_from:
        src = read_weights(args.warm_from)
        ok = {k: v.to(state[k].dtype) for k, v in src.items() if k in state and v.shape == state[k].shape}
        draft.load_state_dict(ok, strict=False)
        loaded = set(ok)
        print(f"warm start: {len(ok)} of {len(state)} tensors from {args.warm_from}. Check that source's license before sharing the result.", flush=True)
    if args.resume:
        draft.load_state_dict({k: v.to(state[k].dtype) for k, v in load_file(args.resume).items()}, strict=True)
        loaded = {n for n, _ in draft.named_parameters() if not n.startswith("injection.")}
        print("resumed", args.resume, flush=True)

    head, embed = target_modules(cfg, dev)
    model = online_model(cfg, draft, head, embed, args.anchors).to(dev)

    slow, fast = [], []
    for n, p in draft.named_parameters():
        if p.requires_grad:
            (slow if n in loaded else fast).append(p)
    backbone_lr = args.lr if args.backbone_lr is None else args.backbone_lr
    groups = [g for g in ({"params": slow, "lr": backbone_lr}, {"params": fast, "lr": args.lr}) if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=0)
    base = [g["lr"] for g in opt.param_groups]
    warm = max(1, int(0.04 * args.steps))

    def lr_at(s):
        return (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, args.steps - warm)))

    def training_files():
        return [f for f in shard_files(args.data) if not is_heldout(f, args.holdout_every)]

    files = training_files()
    if not files:
        raise SystemExit(f"no shards under {args.data}")
    print(len(files), "training shards", flush=True)

    def batch():
        picks = [load_shard(random.choice(files), assistant, ends) for _ in range(args.batch)]
        T = max(len(p[0]) for p in picks)
        width = picks[0][1].shape[1]
        final = picks[0][2].shape[1]
        ids = torch.zeros(len(picks), T, dtype=torch.long)
        hs = torch.zeros(len(picks), T, width, dtype=torch.bfloat16)
        last = torch.zeros(len(picks), T, final, dtype=torch.bfloat16)
        mask = torch.zeros(len(picks), T)
        for i, (a_, b_, c_, m_) in enumerate(picks):
            n = len(a_)
            ids[i, :n], hs[i, :n], last[i, :n], mask[i, :n] = a_, b_.to(torch.bfloat16), c_.to(torch.bfloat16), m_
        return ids.to(dev), hs.to(dev), last.to(dev), mask.to(dev)

    log = open(f"{args.out}/train.jsonl", "a")
    ema = None
    params = slow + fast
    for step in range(args.steps):
        started = time.time()
        if step % 100 == 0:
            files = training_files()              # picks up shards added while training
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * lr_at(step)
        ids, hs, last, mask = batch()
        with torch.autocast(dev, dtype=torch.bfloat16):
            loss, acc, _ = model(input_ids=ids, hidden_states=hs, loss_mask=mask, target_last_hidden_states=last,
                                 max_valid_anchors=args.anchors, global_step=step, total_steps=args.steps)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        value = float(acc.detach())
        ema = value if ema is None else 0.98 * ema + 0.02 * value
        row = {"step": step, "loss": round(float(loss.detach()), 4), "acc": round(value, 4), "acc_ema": round(ema, 4),
               "grad": round(float(grad), 3), "step_s": round(time.time() - started, 2)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if step % 10 == 0:
            print(row, flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            sd = {k: v.detach().cpu().contiguous() for k, v in draft.state_dict().items()}
            tmp = f"{args.out}/.tmp.safetensors"
            save_file(sd, tmp)
            os.replace(tmp, f"{args.out}/model-{step + 1:06d}.safetensors")
            print("saved", step + 1, flush=True)


if __name__ == "__main__":
    main()
