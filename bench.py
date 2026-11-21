import sys, time, torch, torch.nn.functional as F
sys.path.insert(0, '/home/claude/exp/code')
from common import *
torch.set_num_threads(2)
print('torch', torch.__version__)
for res in (224, 160):
    base = build_backbone(); lower = lower_of(base); top = Top(base); top.set_mode(True)
    opt = torch.optim.Adam(top.parameters(), 1e-5)
    x = torch.rand(32, 3, res, res)
    with torch.no_grad():
        lower(normalize(x)); t = time.time()
        for _ in range(3): z = lower(normalize(x))
        tl = (time.time() - t) / 3
    y = torch.randint(0, 6, (32,)); t = time.time()
    for _ in range(3):
        loss = F.cross_entropy(top(z), y); opt.zero_grad(); loss.backward(); opt.step()
    tt = (time.time() - t) / 3
    print(res, f'lower fwd/32imgs {tl:.2f}s  top fwd+bwd/32imgs {tt:.2f}s  -> per epoch(1768 imgs) lower {tl*55:.0f}s top {tt*55:.0f}s')
