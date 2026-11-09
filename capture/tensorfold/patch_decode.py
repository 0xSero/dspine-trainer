"""Adds hidden-state capture to TensorFold's GLM prefill. Applied at image build; refuses to run on unexpected source."""
import sys
p = sys.argv[1]
s = open(p).read()
HELP = '''
_CAP_DIR = __import__("os").environ.get("TF_GLM_CAPTURE_DIR")
_CAP_ALL = __import__("os").environ.get("TF_GLM_CAPTURE_ALL") == "1"
_CAP_BOOT = int(__import__("time").time())          # unique per boot: the counter below restarts at 0
_cap = {"tokens": [], "taps": [], "final": [], "n": 0}


def _cap_rank(e) -> int:
    r = getattr(e, "rank", None)
    if r is None:
        r = getattr(e.w, "rank", None)
    if r is None:
        import torch.distributed as d
        r = d.get_rank() if d.is_initialized() else 0
    return int(r)


def _cap_add(e, b, chunk, R) -> None:
    if not _CAP_DIR or (not _CAP_ALL and _cap_rank(e) != 0):
        return
    _cap["tokens"].extend(int(t) for t in chunk)
    _cap["taps"].append(e.tap_rows(R, b).float().cpu().to(torch.bfloat16))
    _cap["final"].append(b.fnormed[:R].float().cpu().to(torch.bfloat16))


def _cap_flush(e) -> None:
    if not _CAP_DIR or not _cap["taps"]:
        return
    import os
    from safetensors.torch import save_file
    rank = _cap_rank(e)
    d = f"{_CAP_DIR}/rank{rank}"
    os.makedirs(d, exist_ok=True)
    n = _cap["n"]
    tmp = f"{d}/.tmp-{os.getpid()}"
    save_file({"input_ids": torch.tensor(_cap["tokens"], dtype=torch.int32),
               "hidden_states": torch.cat(_cap["taps"]),
               "target_last_hidden_states": torch.cat(_cap["final"])}, tmp)
    os.chmod(tmp, 0o644)
    os.replace(tmp, f"{d}/cap-{_CAP_BOOT}-{n:09d}.safetensors")
    _cap.update(tokens=[], taps=[], final=[], n=n + 1)


'''
anchor = "@torch.no_grad()\ndef prefill(e: Engine, prompt: Sequence[int]"
if s.count(anchor) != 1:
    anchor = "def prefill(e: Engine, prompt: Sequence[int]"
assert s.count(anchor) == 1
s = s.replace(anchor, HELP.lstrip("\n") + anchor)
a = '''    if not prompt:
        raise ValueError("prefill requires at least one token")
'''
assert s.count(a) == 1
s = s.replace(a, a + '    if _CAP_DIR:                                         # capture needs every row: never resume from a cache\n        resume = None\n')
a = '''        prof.report(len(prompt) - begin)
'''
assert s.count(a) == 1
s = s.replace(a, a + '        _cap_flush(e)\n')
a = '''    if drafter is not None:
        drafter.add_taps(e.tap_rows(R, b))
    if mtp:
'''
assert s.count(a) == 1
s = s.replace(a, '''    if drafter is not None:
        _cap_add(e, b, chunk, R)
        drafter.add_taps(e.tap_rows(R, b))
    if mtp:
''')
open(p, "w").write(s)
print("patched", p)
