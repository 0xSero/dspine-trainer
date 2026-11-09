# Capturing hidden states

Training does not run the target model. It reads shards saved while the target prefills your sequences.

## Shard format

One `.safetensors` file per sequence, with any file name, anywhere under `--data`:

| Key | Shape | Type | Meaning |
|---|---|---|---|
| `input_ids` | `[T]` | int | the tokens |
| `hidden_states` | `[T, 5 * hidden]` | bfloat16 | states after the target layers in `draft.dflash_config.target_layer_ids`, concatenated |
| `target_last_hidden_states` | `[T, hidden]` | bfloat16 | the state after the final norm, which the lm_head reads |
| `loss_mask` (optional) | `[T]` | any | 1 where a token should be trained on; default is assistant turns |

For GLM-5.3-Flash a token costs 48 KiB, so 13M tokens take about 620 GB. Any server that can write this format works. SpecForge's SGLang
capture plugin may be a starting point, but it writes its own format and we have not tried adapting it.

## TensorFold patch

`tensorfold/patch_decode.py` adds capture to TensorFold's GLM prefill. It was written against a TensorFold GLM build based on `bf5f945` and stops
with an assertion if the source differs.

```dockerfile
FROM your-tensorfold-glm-image
COPY patch_decode.py /tmp/patch_decode.py
RUN D=/usr/local/lib/python3.12/dist-packages/tensorfold/families/glm5_next/cuda; \
    python3 /tmp/patch_decode.py $D/decode.py && rm -f $D/__pycache__/decode*.pyc /tmp/patch_decode.py
```

Start the server with `PARALLEL=1` and `TF_GLM_CAPTURE_DIR=/path/to/shards`. A drafter must be configured, because the patch runs in the
code path that passes target states to it. The drafter does not affect what is saved. In a multi-GPU setup only rank 0 writes,
unless you set `TF_GLM_CAPTURE_ALL=1`. Then run:

```bash
python -m dspine_trainer.capture_client --data data/train.jsonl --offset-file data/offset --tokens 8000000
```

The server also captures its own startup requests. These are short and contain only fixed test tokens. Our runs left them in without
problems, and you can delete them if you prefer.
