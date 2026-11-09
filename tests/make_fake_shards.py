"""Write small random shards, to check the training loop without a capture server."""
import os
import random
import sys

import torch
from safetensors.torch import save_file

out, count = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40
os.makedirs(out, exist_ok=True)
for i in range(count):
    T = random.randint(300, 900)
    ids = torch.randint(0, 154000, (T,), dtype=torch.int32)
    ids[20] = 154828
    save_file({"input_ids": ids,
               "hidden_states": torch.randn(T, 20480, dtype=torch.bfloat16),
               "target_last_hidden_states": torch.randn(T, 4096, dtype=torch.bfloat16)}, f"{out}/shard-{i:06d}.safetensors")
