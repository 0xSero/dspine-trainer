"""Offline acceptance on held-out shards: how many drafted tokens match the target's greedy next token, in order.

Teacher-forced along each sequence, so it approximates live acceptance. Use it to compare checkpoints with each other.
"""
import argparse
import json
import random

import torch
from safetensors.torch import load_file

from .common import draft_model, is_heldout, load_config, load_shard, online_model, shard_files, target_modules, token_ids


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--config", default="configs/glm-5.3-flash.json")
    a.add_argument("--data", required=True)
    a.add_argument("--ckpt", default="", help="checkpoint to score; empty scores an untrained model")
    a.add_argument("--seqs", type=int, default=120)
    a.add_argument("--anchors", type=int, default=48)
    a.add_argument("--holdout-every", type=int, default=20, help="must match the value used for training")
    a.add_argument("--list", default="", help="file of shard paths relative to --data; keeps the eval set fixed as data grows")
    a.add_argument("--out", default="")
    a.add_argument("--device", default="cuda")
    args = a.parse_args()
    random.seed(7)
    torch.manual_seed(7)
    dev = args.device
    cfg = load_config(args.config)
    assistant, ends = token_ids(cfg)
    draft = draft_model(cfg, dev)
    if args.ckpt:
        state = draft.state_dict()
        draft.load_state_dict({k: v.to(state[k].dtype) for k, v in load_file(args.ckpt).items()}, strict=True)
    head, embed = target_modules(cfg, dev)
    model = online_model(cfg, draft, head, embed, args.anchors).to(dev).eval()
    block = cfg["block_size"]

    if args.list:
        files = [f"{args.data}/{n.strip()}" for n in open(args.list) if n.strip()][: args.seqs]
    else:
        files = [f for f in shard_files(args.data) if is_heldout(f, args.holdout_every)]
        random.shuffle(files)
        files = files[: args.seqs]
    if not files:
        raise SystemExit("no held-out shards; use --list or capture more data")

    pos = {"backbone": torch.zeros(block - 1), "transition": torch.zeros(block - 1)}
    hist = {k: torch.zeros(block) for k in pos}
    n = 0
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
        for path in files:
            ids, hs, last, mask = load_shard(path, assistant, ends)
            ids, hs, last, mask = ids[None].to(dev), hs.to(dev, torch.bfloat16)[None], last.to(dev, torch.bfloat16)[None], mask[None].to(dev)
            length = ids.shape[1]
            if length < 64 or mask.sum() < 40:
                continue
            anchors, keep = model._sample_anchor_positions(length, mask, dev, args.anchors)
            idx = anchors.unsqueeze(-1) + torch.arange(block, device=dev)
            valid = (idx < length).all(-1) & keep
            if valid.sum() == 0:
                continue
            labels = ids.unsqueeze(1).expand(-1, anchors.shape[1], -1).gather(2, idx.clamp(max=length - 1))
            position = torch.cat((torch.arange(length, device=dev).expand(1, -1), model._create_position_ids(anchors)), dim=1)
            out = draft(position_ids=position, attention_mask=model._attention_mask(anchors, keep, length),
                        noise_embedding=model._create_noise_embed(ids, anchors, keep), target_hidden=hs,
                        block_token_ids=labels, replace_mask=None)
            B = valid[0].nonzero().squeeze(-1)
            reference = idx[0, B, 1:] - 1
            greedy = head(last[0][reference.reshape(-1)]).float().argmax(-1).reshape(reference.shape)
            logits = head(out.hidden_states[0, B, 1:]).float()
            candidates = logits.topk(cfg["draft"]["dspine_config"]["top_k"], dim=-1).indices
            scores = draft.transition_scores(out.pre_injection[0, B, 1:], candidates, labels[0, B, 0], head.weight)
            for name, pred in (("backbone", logits.argmax(-1)), ("transition", draft.select_cached(scores, candidates))):
                hit = (pred == greedy).int()
                pos[name] += hit.float().sum(0).cpu()
                for r in hit.cumprod(1).sum(1).cpu().tolist():
                    hist[name][int(r)] += 1
            n += len(B)
    res = {"anchors": n, "ckpt": args.ckpt or "untrained"}
    for k in pos:
        h = hist[k]
        res[k] = {"mean_accepted": round(float((h * torch.arange(block)).sum() / h.sum()), 3),
                  "pos_acc": [round(float(x) / max(n, 1), 3) for x in pos[k]]}
    print(json.dumps(res), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
