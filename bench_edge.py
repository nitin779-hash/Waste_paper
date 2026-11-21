"""Model size / MACs / CPU latency of the full classifier (backbone + head). NOT an embedded-device measurement."""
import json, platform, statistics, sys, time
import torch
sys.path.insert(0, "/home/claude/exp/code")
from common import *
from thop import profile

base = build_backbone()
top = Top(base)
lower = lower_of(base)
model = torch.nn.Sequential(lower, top).eval()
x = torch.rand(1, 3, 224, 224)
macs, params = profile(model, inputs=(normalize(x),), verbose=False)
size_mb = sum(p.numel() * 4 for p in model.parameters()) / 1e6
cpu = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
res = dict(params_m=params / 1e6, macs_m=macs / 1e6, fp32_size_mb=size_mb, cpu=cpu, python=platform.python_version(),
           torch=torch.__version__, latency={})
for th in (1, 2):
    torch.set_num_threads(th)
    with torch.no_grad():
        for _ in range(30):
            model(normalize(x))
        ts = []
        for _ in range(200):
            t = time.perf_counter(); model(normalize(x)); ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    res["latency"][f"{th}_thread"] = dict(median_ms=statistics.median(ts), p95_ms=ts[int(0.95 * len(ts))],
                                          fps=1000 / statistics.median(ts))
json.dump(res, open(f"{OUT}/edge_bench.json", "w"), indent=1)
print(json.dumps(res, indent=1))
