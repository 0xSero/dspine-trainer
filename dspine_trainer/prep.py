"""Build the tokenized dataset that capture sends through the target model.

Sources are Hugging Face datasets or local JSONL files, in either chat layout:
  {"conversations": [{"from": "human", "value": "..."}, ...]}
  {"messages": [{"role": "user", "content": "..."}, ...]}

  python -m dspine_trainer.prep --source mlabonne/open-perfectblend --tokens 60000000 --out data/train.jsonl
  python -m dspine_trainer.prep --source mine.jsonl --out data/train.jsonl
  python -m dspine_trainer.prep --source repo/a@0.7 --source mine.jsonl@0.3 --out data/train.jsonl
"""
import argparse
import json
import os
import random

from datasets import load_dataset

from .common import load_config, supervised_mask, token_ids, tokenizer

ROLES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system",
         "tool": "tool", "observation": "tool", "function_response": "tool"}


def messages(row):
    turns = row.get("conversations") or row.get("messages")
    if not turns:
        return None
    out = []
    for m in turns:
        role = ROLES.get(m.get("from") or m.get("role"))
        content = m.get("value", m.get("content"))
        if role is None or not isinstance(content, str):
            return None
        out.append({"role": role, "content": content})
    return out if any(m["role"] == "assistant" for m in out) else None


def read(spec, split, config):
    if os.path.exists(spec):
        return load_dataset("json", data_files=spec, split="train")
    return load_dataset(spec, config, split=split)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--config", default="configs/glm-5.3-flash.json")
    a.add_argument("--source", action="append", required=True, help="dataset id or JSONL path, optionally @weight")
    a.add_argument("--split", default="train")
    a.add_argument("--hf-config", default=None, help="dataset config name, if the dataset has several")
    a.add_argument("--tokens", type=int, default=60_000_000, help="total tokens to keep across sources")
    a.add_argument("--max-length", type=int, default=3072)
    a.add_argument("--min-length", type=int, default=1000, help="capture and training cost per sequence, so short ones are poor value")
    a.add_argument("--min-supervised", type=int, default=32)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--out", required=True)
    args = a.parse_args()
    cfg = load_config(args.config)
    tok = tokenizer(cfg)
    assistant, ends = token_ids(cfg)

    specs = [s.rsplit("@", 1) if "@" in s else [s, "1"] for s in args.source]
    total_weight = sum(float(w) for _, w in specs)
    rows = []
    for spec, weight in specs:
        budget = int(args.tokens * float(weight) / total_weight)
        ds = read(spec, args.split, args.hf_config)
        order = list(range(len(ds)))
        random.Random(args.seed).shuffle(order)
        n = kept = 0
        for i in order:
            msgs = messages(ds[int(i)])
            if msgs is None:
                continue
            try:
                ids = list(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_dict=False))
            except Exception:
                continue
            ids = ids[: args.max_length]
            if len(ids) < args.min_length or sum(supervised_mask(ids, assistant, ends)) < args.min_supervised:
                continue
            rows.append(ids)
            n += 1
            kept += len(ids)
            if kept >= budget:
                break
        print(f"{spec}: {n} sequences, {kept} tokens", flush=True)
    random.Random(args.seed).shuffle(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for ids in rows:
            f.write(json.dumps({"tokens": ids}) + "\n")
    print(f"wrote {len(rows)} sequences to {args.out}", flush=True)


if __name__ == "__main__":
    main()
