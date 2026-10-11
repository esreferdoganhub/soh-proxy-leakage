"""
scripts/mit_tuned_benchmark.py
mit_benchmark.py'nin EŞİT AYAR BÜTÇELİ ve KALDIĞI YERDEN DEVAM EDEBİLEN sürümü
(Colab kesintilerine dayanıklı).

Protokol
1) Ayar (tune): her (özellik kümesi, model) için önceden tanımlı ızgaradan K aday.
   Seçim ölçütü: tohum 0'ın TUNE_FOLDS katlarındaki DOĞRULAMA hücrelerinde ortalama
   hücre-makro MAE. Test hücreleri bu aşamada hiç kullanılmaz.
2) Son (final): seçilen ayar her tohum × kat için yeniden eğitilir, test bir kez ölçülür.
Her model için aday sayısı aynıdır (K) → eşit ayar bütçesi.

Ek düzeltmeler (tüm modellere eşit):
- Ölçeklenmiş girdiler eğitim dağılımının ±CLIP standart sapmasına kırpılır
  (düzleştirilmiş Ridge'in birkaç hücrede dışdeğerlemeyle patlamasını önler).
- 'transformer': HAT-Net omurgasıyla aynı RoPE kodlayıcı, ama son adım havuzlama +
  doğrusal çıkış (HAT-Net ortalama havuzlama + sigmoid + BN/Dropout kafa kullanır).

Kullanım:
  python scripts/mit_tuned_benchmark.py --out results/mit_tuned           # tam
  python scripts/mit_tuned_benchmark.py --features ops --models ridge --K 2 --seeds 0 --out /tmp/x  # duman
"""

import argparse
import copy
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import mit_benchmark as mb  # noqa: E402

CLIP = 5.0
TUNE_FOLDS = (0, 1)

GRIDS = {
    "mean": {"view": ["flat"]},  # eğitim ortalaması (ayarsız referans)
    "ridge": {"alpha": [1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4], "view": ["flat"]},
    "last": {"alpha": [1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4], "view": ["last"]},
    "hgb": {"max_iter": [200, 500], "learning_rate": [0.03, 0.1], "max_leaf_nodes": [15, 63],
            "l2_regularization": [0.0, 1.0], "view": ["flat"]},
    "lstm": {"hidden": [32, 64, 128], "layers": [1, 2], "lr": [3e-4, 1e-3, 3e-3], "dropout": [0.0, 0.2]},
    "transformer": {"d_model": [32, 64, 128], "layers": [1, 2], "lr": [3e-4, 1e-3], "dropout": [0.0, 0.1]},
    "hat_net": {"lr": [1e-4, 3e-4, 1e-3, 3e-3], "weight_decay": [1e-4, 1e-2], "batch": [128, 512]},
}
TORCH_MODELS = {"lstm", "transformer", "hat_net"}


def sample_configs(model: str, K: int, seed: int = 123) -> list[dict]:
    g = GRIDS[model]
    allc = [dict(zip(g, v)) for v in itertools.product(*g.values())]
    if len(allc) <= K:
        return allc
    return random.Random(f"{model}-{seed}").sample(allc, K)


class ClipScaler(mb.Scaler):
    def __call__(self, X):
        return np.clip(super().__call__(X), -CLIP, CLIP).astype(np.float32)


# ───────────────────────────── modeller ─────────────────────────────
class LSTMReg(torch.nn.Module):
    def __init__(self, nf, hidden, layers, dropout):
        super().__init__()
        self.lstm = torch.nn.LSTM(nf, hidden, num_layers=layers, batch_first=True,
                                  dropout=dropout if layers > 1 else 0.0)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


class TransformerReg(torch.nn.Module):
    def __init__(self, nf, d_model, layers, dropout):
        super().__init__()
        from model import RoPETransformerEncoder

        self.inp = torch.nn.Linear(nf, d_model)
        self.enc = RoPETransformerEncoder(d_model, n_heads=4, n_layers=layers, dim_ff=2 * d_model, dropout=dropout)
        self.head = torch.nn.Linear(d_model, 1)

    def forward(self, x):
        return self.head(self.enc(self.inp(x))[:, -1]).squeeze(-1)


class HATWrap(torch.nn.Module):
    def __init__(self, nf, T):
        super().__init__()
        from model import HATNet

        self.m = HATNet(n_features=nf, seq_len=T)

    def forward(self, x):
        return self.m(x)[0]


def fit_sklearn(model, cfg, Xtr, ytr, seed):
    f = (lambda X: X[:, -1, :]) if cfg.get("view") == "last" else (lambda X: X.reshape(len(X), -1))
    if model == "mean":
        mu = float(ytr.mean())
        return lambda X: np.full(len(X), mu), {"epoch": -1}
    if model in ("ridge", "last"):
        m = Ridge(alpha=cfg["alpha"])
    else:
        m = HistGradientBoostingRegressor(max_iter=cfg["max_iter"], learning_rate=cfg["learning_rate"],
                                          max_leaf_nodes=cfg["max_leaf_nodes"],
                                          l2_regularization=cfg["l2_regularization"], random_state=seed)
    m.fit(f(Xtr), ytr)
    return lambda X: m.predict(f(X)), {"epoch": -1}


def fit_torch(model, cfg, Xtr, ytr, Xva, yva, cva, seed, epochs, patience, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    nf, T = Xtr.shape[-1], Xtr.shape[1]
    if model == "lstm":
        net = LSTMReg(nf, cfg["hidden"], cfg["layers"], cfg["dropout"])
    elif model == "transformer":
        net = TransformerReg(nf, cfg["d_model"], cfg["layers"], cfg["dropout"])
    else:
        net = HATWrap(nf, T)
    net = net.to(device)
    if model == "hat_net":  # sigmoid çıkış → [0,1] min-max (özgün tasarım korunur)
        lo, hi = ytr.min() - 0.05, ytr.max() + 0.05
        enc, dec = (lambda y: (y - lo) / (hi - lo)), (lambda z: z * (hi - lo) + lo)
    else:
        mu, sd = ytr.mean(), ytr.std() + 1e-8
        enc, dec = (lambda y: (y - mu) / sd), (lambda z: z * sd + mu)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    g = torch.Generator().manual_seed(seed)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(enc(ytr).astype(np.float32)))
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.get("batch", 256), shuffle=True, generator=g,
                                     drop_last=len(ds) > cfg.get("batch", 256))

    def predict(X):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):
                out.append(net(torch.from_numpy(X[i : i + 4096]).to(device)).float().cpu().numpy())
        return dec(np.concatenate(out))

    best, wait = (np.inf, 0, None), 0
    for ep in range(1, epochs + 1):
        net.train()
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(net(xb.to(device)), yb.to(device)).backward()
            opt.step()
        s = mb.macro_mae(yva, predict(Xva), cva)
        if s < best[0]:
            best, wait = (s, ep, copy.deepcopy(net.state_dict())), 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best[2])
    return predict, {"epoch": best[1]}


# ───────────────────────────── çalıştırma ─────────────────────────────
class Ledger:
    """Her eğitimin sonucunu CSV'ye ekler; yeniden başlatılınca yapılanları atlar."""

    def __init__(self, path: Path):
        self.path = path
        self.done = set()
        if path.exists():
            d = pd.read_csv(path)
            self.done = set(zip(d.phase, d.features, d.model, d.cfg_id, d.seed, d.fold))

    def has(self, key):
        return key in self.done

    def add(self, rows: list[dict]):
        df = pd.DataFrame(rows)
        df.to_csv(self.path, mode="a", header=not self.path.exists(), index=False)
        r = rows[0]
        self.done.add((r["phase"], r["features"], r["model"], r["cfg_id"], r["seed"], r["fold"]))


def fold_data(df, cols, seed, k, folds, T, stride, cache):
    key = (seed, k)
    if key not in cache:
        cells = np.array(sorted(df.cell.unique()))
        perm = np.random.default_rng(seed).permutation(cells)
        tr_idx, te_idx = list(GroupKFold(folds).split(perm, groups=perm))[k]
        tr, te = perm[tr_idx], perm[te_idx]
        n_val = max(1, len(tr) // 5)
        va, tr = tr[:n_val], tr[n_val:]
        Xtr, ytr, _ = mb.make_windows(df, cols, tr, T, stride)
        Xva, yva, cva = mb.make_windows(df, cols, va, T, stride)
        Xte, yte, cte = mb.make_windows(df, cols, te, T, stride)
        sc = ClipScaler().fit(Xtr)
        cache[key] = (sc(Xtr), ytr, sc(Xva), yva, cva, sc(Xte), yte, cte)
    return cache[key]


def run_one(model, cfg, data, seed, a, device):
    Xtr, ytr, Xva, yva, cva, Xte, yte, cte = data
    t0 = time.time()
    if model in TORCH_MODELS:
        pred, info = fit_torch(model, cfg, Xtr, ytr, Xva, yva, cva, seed, a.epochs, a.patience, device)
    else:
        pred, info = fit_sklearn(model, cfg, Xtr, ytr, seed)
    val = mb.macro_mae(yva, pred(Xva), cva)
    return pred, val, info, time.time() - t0


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"cihaz: {device}", flush=True)
    tune_led, final_led = Ledger(out / "tune_log.csv"), Ledger(out / "final_per_cell.csv")
    chosen = json.load(open(out / "chosen.json")) if (out / "chosen.json").exists() else {}

    for fs in a.features:
        df, cols = mb.load_frame(fs)
        cache = {}
        print(f"\n=== {fs}: {df.cell.nunique()} hücre, {len(cols)} özellik", flush=True)
        for model in a.models:
            cfgs = sample_configs(model, a.K)
            # ── 1) ayar ──
            for ci, cfg in enumerate(cfgs):
                for k in TUNE_FOLDS:
                    key = ("tune", fs, model, ci, 0, k)
                    if tune_led.has(key):
                        continue
                    _, val, info, dt = run_one(model, cfg, fold_data(df, cols, 0, k, a.folds, a.T, a.stride, cache), 0, a, device)
                    tune_led.add([{"phase": "tune", "features": fs, "model": model, "cfg_id": ci, "seed": 0, "fold": k,
                                   "cfg": json.dumps(cfg), "val_macro_mae": val, "secs": dt, **info}])
                    print(f"  tune {model:11s} cfg{ci} fold{k} val={val:.4f} ({dt:.0f}s)", flush=True)
            t = pd.read_csv(out / "tune_log.csv")
            t = t[(t.features == fs) & (t.model == model)]
            best_ci = int(t.groupby("cfg_id").val_macro_mae.mean().idxmin())
            chosen[f"{fs}|{model}"] = {"cfg_id": best_ci, "cfg": cfgs[best_ci],
                                       "val_macro_mae": float(t.groupby("cfg_id").val_macro_mae.mean().min())}
            json.dump(chosen, open(out / "chosen.json", "w"), indent=1)
            print(f"  ✔ {model} seçildi: {cfgs[best_ci]}", flush=True)
            # ── 2) son ──
            for seed in a.seeds:
                cache_s = cache if seed == 0 else {}
                for k in range(a.folds):
                    key = ("final", fs, model, best_ci, seed, k)
                    if final_led.has(key):
                        continue
                    data = fold_data(df, cols, seed, k, a.folds, a.T, a.stride, cache_s)
                    pred, val, info, dt = run_one(model, cfgs[best_ci], data, seed, a, device)
                    pc = mb.per_cell(data[6], pred(data[5]), data[7])
                    final_led.add([{"phase": "final", "features": fs, "model": model, "cfg_id": best_ci, "seed": seed,
                                    "fold": k, "cell": c, **v, "secs": dt} for c, v in pc.items()])
                    print(f"  final {model:11s} seed{seed} fold{k} test={np.mean([v['mae'] for v in pc.values()]):.4f} ({dt:.0f}s)", flush=True)

    res = pd.read_csv(out / "final_per_cell.csv")
    summ, tests = mb.summarize(res)
    summ.to_csv(out / "summary.csv", index=False)
    tests.to_csv(out / "wilcoxon.csv", index=False)
    print("\n", summ.round(4).to_string(index=False))
    print("\n", tests.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=["ops", "cycle", "curve@3.2", "curve@3.1", "curve@3.0", "curve@2.0", "leaky"])
    ap.add_argument("--models", nargs="+", default=["mean", "last", "ridge", "hgb", "lstm", "transformer", "hat_net"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results" / "mit_tuned"))
    main(ap.parse_args())
