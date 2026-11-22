import torch, time
from common import *
import aug
m = load_model(); X = load_forget(); out = {}; t = time.time()
for name, fn in aug.probe_bank(0):
    zs = []
    for i in range(0, 3500, 500):
        zs.append(predict(m, fn(X[i:i+500]), bs=50))
    out[name] = torch.cat(zs); print(name, f"{time.time()-t:.0f}s", flush=True)
torch.save(out, OUT/"eda"/"probe_logits.pt")
