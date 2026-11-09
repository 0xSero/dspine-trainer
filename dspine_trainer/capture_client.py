"""Send tokenized sequences through a capturing server. Resumes from an offset file.

The server must save one shard per request; see capture/README.md. Only prefill is needed, so max_tokens is 1.
"""
import argparse
import json
import os
import shutil
import time
import urllib.request


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--url", default="http://127.0.0.1:8888")
    a.add_argument("--data", required=True)
    a.add_argument("--offset-file", required=True)
    a.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    a.add_argument("--tokens", type=int, default=8_000_000, help="stop after this many tokens")
    a.add_argument("--grid", type=int, default=64, help="trim prompts to a multiple of this; TensorFold prefills only on its grid")
    a.add_argument("--min-free-gib", type=int, default=300, help="stop when the disk holding the data has less free space")
    args = a.parse_args()
    start = int(open(args.offset_file).read()) if os.path.exists(args.offset_file) else 0
    done = tokens = 0
    started = time.time()
    with open(args.data) as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            ids = json.loads(line)["tokens"]
            ids = ids[: len(ids) // args.grid * args.grid]
            body = json.dumps({"model": args.model, "prompt": ids, "max_tokens": 1, "temperature": 0}).encode()
            req = urllib.request.Request(args.url + "/v1/completions", body, {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=600).read()
            done += 1
            tokens += len(ids)
            with open(args.offset_file + ".tmp", "w") as o:
                o.write(str(i + 1))
            os.replace(args.offset_file + ".tmp", args.offset_file)
            if done % 20 == 0 or done == 1:
                print(f"{done} sequences, {tokens} tokens, {tokens / (time.time() - started):.0f} tok/s", flush=True)
            if tokens >= args.tokens or shutil.disk_usage(os.path.dirname(os.path.abspath(args.data))).free < args.min_free_gib * 2**30:
                break
    print(f"done: {done} sequences, {tokens} tokens", flush=True)


if __name__ == "__main__":
    main()
