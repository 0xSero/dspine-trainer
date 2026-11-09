# DSpine Trainer

Train a [DSpine](https://arxiv.org/abs/2609.36173) speculative-decoding drafter on your own data. The included config targets
GLM-5.3-Flash, and the same steps should apply to another target if you write a config for it.

A drafter guesses several tokens ahead. The target model checks them all in one step, so output is still exactly the target's. A better
drafter gets more tokens accepted per step.

This repository contains code only, with no weights or datasets. The model and loss come from
[SpecForge](https://github.com/sgl-project/SpecForge).

## How it works

1. **Prepare:** tokenize conversations with the target's chat template.
2. **Capture:** run them through the target and save its hidden states.
3. **Train:** fit the drafter on those shards.
4. **Evaluate:** score checkpoints on held-out shards.

## Install

```bash
git clone https://github.com/0xSero/dspine-trainer && cd dspine-trainer
uv sync
```

On an NVIDIA container with its own PyTorch build, make a venv that uses the system packages and do not install torch into it:

```bash
uv venv --system-site-packages .venv
uv pip install --no-deps "git+https://github.com/sgl-project/SpecForge.git@2dadea10e317842af21c79e2097dc05abe1e2b5a"
uv pip install datasets huggingface-hub safetensors transformers==5.12.1
```

## Quick start

```bash
# 1. Prepare the default dataset
uv run python -m dspine_trainer.prep --source mlabonne/open-perfectblend --tokens 60000000 --out data/train.jsonl

# 2. Capture: see capture/README.md
uv run python -m dspine_trainer.capture_client --data data/train.jsonl --offset-file data/offset

# 3. Train
uv run python -m dspine_trainer.train --data shards --out runs/first

# 4. Evaluate
uv run python -m dspine_trainer.evaluate --data shards --ckpt runs/first/model-003000.safetensors
```

To check your install without a capture server, run `python tests/make_fake_shards.py /tmp/fake` and then train on `/tmp/fake` with
`--steps 2 --anchors 16`. Random shards only prove that the loop runs.

## Bring your own data

Pass a JSONL file with one conversation per line, in either layout:

```json
{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

```bash
python -m dspine_trainer.prep --source mine.jsonl --out data/train.jsonl
python -m dspine_trainer.prep --source mine.jsonl@0.6 --source mlabonne/open-perfectblend@0.4 --out data/train.jsonl
```

`@weight` sets each source's share of `--tokens`. Only supervised tokens, by default those inside assistant turns, are trained on. Files you
pass stay on your machine, and nothing in this code uploads them. Be careful sharing a drafter trained on private data.

## Which data to use

These licenses and layouts were checked on Hugging Face in November 2026. Only Open-PerfectBlend has been trained on with this code.

| Dataset | License | Good for | Use it with |
|---|---|---|---|
| [`mlabonne/open-perfectblend`](https://huggingface.co/datasets/mlabonne/open-perfectblend) | Apache-2.0 | General default | no extra flags |
| [`NousResearch/hermes-function-calling-v1`](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) | Apache-2.0 | Tool calls, JSON | `--hf-config func_calling_singleturn --min-length 1` |
| [`open-r1/OpenR1-Math-220k`](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | Apache-2.0 | Math reasoning | `--min-length 200` |
| [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT | Long multi-turn chat | `--split train_sft` |

A dataset with a different layout, such as `glaiveai/glaive-function-calling-v2` or `nvidia/OpenCodeReasoning`, needs converting to JSONL first.

We do not recommend raw user-chat collections such as WildChat or LMSYS-Chat-1M as a default, because they can contain personal information.

Three things we learned about choosing data:

- **Check the mix after filtering.** The default `--min-length 1000` keeps capture efficient but changes what you train on: Open-PerfectBlend
  came out 78% from one long-chat source and under 0.5% from its two math sets.
- **Include what you will serve.** Our drafter was weakest on structured output, and no source in its mix was dedicated to it. We have not yet
  confirmed that adding some would fix this.
- **Prefer answers written by the target.** The DSpine paper regenerated its training answers with the target model. Open-PerfectBlend's answers
  come from other models and matched GLM-5.3-Flash's own greedy choice only 68% of the time.

## Results

Tested on one NVIDIA DGX Spark for training, and two for capture and serving, with GLM-5.3-Flash as the target.

| Measure | Value |
|---|---:|
| Training speed, batch 4 at 64 anchors | about 8 s per step |
| Capture speed | about 950 tok/s |
| Training at the paper's scale (409K sequences at 512 anchors) | about 75 days on one Spark |

**A short training run does not beat DFlash2.** After about 4,600 steps, which is roughly 0.6% of the paper's training positions, our drafter
accepted 1.31 tokens per block offline. Live, an earlier checkpoint reached 2.13 tokens per round on prose against DFlash2's 3.25.
Starting from random weights is also much slower than warm-starting: 0.28 against 0.99 offline at step 800.

`--warm-from` starts from compatible weights in another checkpoint. It is off by default because the result inherits the source's
license. For example, `incoai/GLM-5.3-Flash-DFlash2` is CC BY-NC-ND, so a drafter warm-started from it cannot be shared publicly.

## Using another target

Copy `configs/glm-5.3-flash.json` and change the target repo and revision, weight names, role tokens, vocabulary size, hidden size, layer count,
mask token and `target_layer_ids`. Your capture must save those same layers. Only GLM-5.3-Flash has been tested.

## License

Apache-2.0. See `NOTICE` for upstream credits. Datasets and model weights have their own licenses.
