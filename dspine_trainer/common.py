"""Shared helpers: config, target weights, shards and the DSpine model."""
import glob
import json
import os
import zlib

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file
from specforge.algorithms.dspine.model import OnlineDSpineModel
from specforge.modeling.draft.dspine import DSpineDraftModel
from transformers import AutoConfig, AutoTokenizer


def load_config(path):
    with open(path) as f:
        return json.load(f)


def tokenizer(cfg):
    return AutoTokenizer.from_pretrained(cfg["target"]["repo"], revision=cfg["target"]["revision"])


def token_ids(cfg):
    """Return (assistant id, set of ids that end an assistant span)."""
    tok = tokenizer(cfg)
    ids = {t: tok.convert_tokens_to_ids(t) for t in [cfg["tokens"]["assistant"], *cfg["tokens"]["end_of_assistant"]]}
    missing = [t for t, i in ids.items() if i is None or i < 0]
    if missing:
        raise ValueError(f"tokens not found in the target tokenizer: {missing}")
    return ids[cfg["tokens"]["assistant"]], {ids[t] for t in cfg["tokens"]["end_of_assistant"]}


def supervised_mask(tokens, assistant, ends):
    """1 for tokens inside assistant turns, 0 elsewhere. The assistant role token itself is not supervised."""
    mask, on = [], False
    for t in tokens:
        if t == assistant:
            on = True
            mask.append(0)
            continue
        if t in ends:
            on = False
        mask.append(1 if on else 0)
    return mask


def draft_model(cfg, device, dtype=torch.bfloat16):
    values = {k: v for k, v in cfg["draft"].items() if k != "architectures"}
    config = AutoConfig.for_model(**values)
    config.architectures = ["DSpineDraftModel"]
    return DSpineDraftModel(config).to(device, dtype=dtype)


def target_tensor(cfg, key):
    t = cfg["target"]
    index = hf_hub_download(t["repo"], "model.safetensors.index.json", revision=t["revision"])
    with open(index) as f:
        name = json.load(f)["weight_map"][key]
    with safe_open(hf_hub_download(t["repo"], name, revision=t["revision"]), "pt") as f:
        return f.get_tensor(key)


def target_modules(cfg, device, dtype=torch.bfloat16):
    """The target's lm_head and token embeddings. Both stay frozen."""
    lm = target_tensor(cfg, cfg["target"]["lm_head"])
    em = target_tensor(cfg, cfg["target"]["embed_tokens"])
    head = torch.nn.Linear(lm.shape[1], lm.shape[0], bias=False).to(device, dtype=dtype)
    head.weight.data.copy_(lm.to(device, dtype=dtype))
    embed = torch.nn.Embedding(em.shape[0], em.shape[1]).to(device, dtype=dtype)
    embed.weight.data.copy_(em.to(device, dtype=dtype))
    return head, embed


def online_model(cfg, draft, head, embed, anchors):
    return OnlineDSpineModel(
        draft_model=draft,
        target_lm_head=head,
        target_embed_tokens=embed,
        mask_token_id=cfg["draft"]["dflash_config"]["mask_token_id"],
        block_size=cfg["block_size"],
        attention_backend="sdpa",
        num_anchors=anchors,
        objective_chunk_blocks=16,
        warmup_ratio=0.04,
    )


def shard_files(data):
    return sorted(glob.glob(os.path.join(data, "**", "*.safetensors"), recursive=True))


def is_heldout(path, every):
    """Stable split by file name, so adding shards never moves an existing one across the split."""
    return every > 0 and zlib.crc32(os.path.basename(path).encode()) % every == 0


def load_shard(path, assistant, ends):
    t = load_file(path)
    ids = t["input_ids"].long()
    mask = t.get("loss_mask")
    if mask is None:
        mask = torch.tensor(supervised_mask(ids.tolist(), assistant, ends))
    return ids, t["hidden_states"], t["target_last_hidden_states"], mask.float()
