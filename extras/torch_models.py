"""The two neural networks of Part 10 (PyTorch; not part of run_selftests.py, which needs only NumPy).

    pip install torch numpy        # used with torch 2.8.0
    python extras/torch_models.py  # self-test, about a minute on a laptop CPU

- MLP: 22 features -> 64 -> 32 -> 1, ReLU, dropout 0.1.
- LSTM: the last 32 fifteen-minute bars of 4 series (BTC and ETH returns and absolute returns) -> hidden size 32 -> 1.
Training: mean squared error, AdamW (learning rate 1e-3, weight decay 1e-4), batches of 2,048, at most 60 epochs, **early stopping on the
validation-period loss** (stop after 6 epochs without improvement and restore the best epoch). Fixed seeds, CPU only, deterministic algorithms:
the same inputs give bit-identical predictions (the self-test checks this).
"""
import sys

import numpy as np
import torch
from torch import nn

SETTINGS = dict(batch=2048, max_epochs=60, patience=6, lr=1e-3, weight_decay=1e-4, dropout=0.1, seed=0)


def _fix(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


class MLP(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


class LSTM(nn.Module):
    def __init__(self, d, hidden=32):
        super().__init__()
        self.rnn = nn.LSTM(d, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(h[:, -1, :]).squeeze(-1)


def train(make, x_train, y_train, x_valid, y_valid, settings=None):
    """-> (model, dict(valid_loss per epoch, best_epoch)). Early stopping on the validation loss; the best epoch's weights are restored."""
    s = dict(SETTINGS, **(settings or {}))
    _fix(s["seed"])
    m = make()
    opt = torch.optim.AdamW(m.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])
    xt, yt = torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    xv, yv = torch.tensor(x_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.float32)
    g = torch.Generator().manual_seed(s["seed"])
    best, best_epoch, best_state, stale, log = float("inf"), -1, None, 0, []
    for epoch in range(s["max_epochs"]):
        m.train()
        order = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), s["batch"]):
            j = order[i:i + s["batch"]]
            opt.zero_grad()
            loss = ((m(xt[j]) - yt[j]) ** 2).mean()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = float(((m(xv) - yv) ** 2).mean())
        log.append(v)
        if v < best - 1e-9:
            best, best_epoch, stale = v, epoch, 0
            best_state = {k: t.clone() for k, t in m.state_dict().items()}
        else:
            stale += 1
            if stale >= s["patience"]:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, dict(valid_loss=log, best_epoch=best_epoch)


def predict(m, x):
    with torch.no_grad():
        return m(torch.tensor(x, dtype=torch.float32)).numpy().astype(np.float64)


def selftest():
    ok = [0, 0]

    def check(name, cond):
        ok[0 if cond else 1] += 1
        print(("  ok    " if cond else "  FAIL  ") + name)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6000, 3)).astype(np.float32)
    y = (x[:, 0] ** 2 - 1 + 0.3 * rng.normal(size=6000)).astype(np.float32)          # U-shaped: invisible to a linear model
    fast = dict(max_epochs=80, batch=256, lr=3e-3)
    m, rec = train(lambda: MLP(3, 0.0), x[:4000], y[:4000], x[4000:5000], y[4000:5000], fast)
    p = predict(m, x[5000:])
    r2 = 1 - np.mean((p - y[5000:]) ** 2) / np.var(y[5000:])
    coef = np.linalg.lstsq(np.c_[np.ones(4000), x[:4000]], y[:4000], rcond=None)[0]
    r2_lin = 1 - np.mean(((np.c_[np.ones(1000), x[5000:]] * coef).sum(axis=1) - y[5000:]) ** 2) / np.var(y[5000:])   # no @: spurious warnings on some Macs
    check(f"the MLP learns a U shape (y = x^2 - 1 + noise): out-of-sample R^2 {r2:.2f} > 0.7, linear regression {r2_lin:.2f}", r2 > 0.7 and r2_lin < 0.05)
    m2, _ = train(lambda: MLP(3, 0.0), x[:4000], y[:4000], x[4000:5000], y[4000:5000], fast)
    check("fixed seeds and deterministic algorithms: training twice gives bit-identical predictions", np.array_equal(predict(m2, x[5000:]), p))
    check("early stopping restores the epoch with the lowest validation loss", rec["valid_loss"][rec["best_epoch"]] == min(rec["valid_loss"]))
    seq = rng.normal(size=(5000, 8, 2)).astype(np.float32)
    ys = (np.sign(seq[:, -1, 0]) + 0.2 * rng.normal(size=5000)).astype(np.float32)   # depends only on the sign of the last step's first channel
    ml, _ = train(lambda: LSTM(2, 16), seq[:3500], ys[:3500], seq[3500:4250], ys[3500:4250], dict(max_epochs=60, batch=256, lr=3e-3))
    acc = float(np.mean(np.sign(predict(ml, seq[4250:])) == np.sign(ys[4250:])))
    check(f"the LSTM learns 'the sign of the last step': out-of-sample hit rate {acc:.0%} > 90 %", acc > 0.9)
    print(f"\n  {ok[0]} passed / {ok[1]} failed")
    return ok[1] == 0


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
