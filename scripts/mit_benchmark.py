"""
scripts/mit_benchmark.py
MIT/Severson (124 LFP hücre) üzerinde sızıntısız, hücre-bazlı SOH karşılaştırması.

Soru: Hücreler arası (cross-cell) değerlendirmede derin modeller (LSTM, HAT-Net)
basit temel modelleri (eğitim ortalaması, Ridge, HGB) geçiyor mu?

Protokol
- Bölme hücre düzeyinde: GroupKFold(5) × tohumlar; her katta eğitim hücrelerinin
  %20'si doğrulama (model seçimi/erken durdurma) için ayrılır. Test hücreleri
  yalnızca seçilen checkpoint ile bir kez değerlendirilir.
- Girdi: son T döngünün özellik penceresi (nedensel; gelecek döngü yok).
- Ölçekleyiciler yalnızca eğitim hücrelerine fit edilir.
- Metrik: hücre başına MAE/RMSE → hücreler üzerinde ortalama (her hücre eşit ağırlık);
  hücre-bazlı bootstrap %95 GA; modeller arası eşleştirilmiş Wilcoxon (hücre başına MAE).

Özellik kümeleri (vekil/özellik sızıntısı ablasyonu):
  ops       : IR, Tavg, Tmax, Tmin, chargetime     (deşarj eğrisi yok — gerçekçi BMS)
  curve@Vc  : ops + deşarj eğrisi ΔQ(V), yalnızca V ≥ Vc noktaları (hücrenin kendi
              10. döngüsüne göre). 4C LFP'de Vc=3.2 V ≈ kapasitenin %21'i,
              3.1 V ≈ %58, 3.0 V ≈ %74, 2.0 V = tam eğri (%100) gözlenir.
  cycle     : ops + döngü sayısı (yaşın doğrudan vekili)
  leaky     : ops + QCharge (r≈0.999 ile QDischarge; Coulomb verimi ~1 → hedefin vekili)

Kullanım:
  python scripts/mit_benchmark.py --features ops curve@3.2 curve@3.1 curve@3.0 curve@2.0 leaky
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import PROC_DIR  # noqa: E402

RESULTS = ROOT / "results" / "mit_untuned"
OPS = ["IR", "Tavg", "Tmax", "Tmin", "chargetime"]
REF_CYCLE = 10


# ───────────────────────────── veri ─────────────────────────────
def load_frame(feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    cyc = pd.read_csv(PROC_DIR / "mit_all_cycles.csv")
    # İlk döngü(ler) Severson verisinde hatalı/eksik; kapasite aykırılarını at
    cyc = cyc[(cyc.row >= 1) & cyc.QDischarge.between(0.5, 1.2)].copy()
    nominal = 1.1
    cyc["soh"] = cyc.QDischarge / nominal
    # Ölçüm hataları: IR=0 (ölçülmemiş), şarj süresi > 60 dk (test duraklaması)
    cyc.loc[cyc.IR <= 0, "IR"] = np.nan
    cyc.loc[cyc.chargetime > 60, "chargetime"] = np.nan
    cols = list(OPS)

    if feature_set == "leaky":
        cols.append("QCharge")
    elif feature_set == "cycle":
        cols.append("row")  # döngü sayısı: yaşla korelasyonlu klasik vekil
    elif feature_set.startswith("curve@"):
        v_cut = float(feature_set.split("@")[1])
        cur = np.load(PROC_DIR / "mit_all_curves.npz")
        v = cur["voltage"]
        keep = np.where(v >= v_cut - 1e-9)[0]
        q = cur["qdlin"][cyc.curve_row.to_numpy()][:, keep]
        dq_cols = [f"dQ_{v[k]:.2f}V" for k in keep]
        dq = pd.DataFrame(q, columns=dq_cols, index=cyc.index)
        cyc = pd.concat([cyc, dq], axis=1)
        # Hücrenin kendi referans döngüsüne göre fark (Severson ΔQ(V) fikri, kısmi pencere)
        ref = cyc[cyc.row == REF_CYCLE].set_index("cell")[dq_cols]
        cyc = cyc[cyc.cell.isin(ref.index) & (cyc.row >= REF_CYCLE)]
        cyc[dq_cols] = cyc[dq_cols].to_numpy() - ref.loc[cyc.cell].to_numpy()
        cols += dq_cols
    elif feature_set != "ops":
        raise ValueError(feature_set)

    cyc = cyc.replace([np.inf, -np.inf], np.nan)
    cyc[cols] = cyc.groupby("cell")[cols].transform(lambda s: s.ffill())
    cyc = cyc.dropna(subset=cols + ["soh"])
    return cyc.sort_values(["cell", "row"]).reset_index(drop=True), cols


def make_windows(df, cols, cells, T, stride):
    X, y, c = [], [], []
    for cell in cells:
        g = df[df.cell == cell]
        A = g[cols].to_numpy(np.float32)
        s = g.soh.to_numpy(np.float32)
        for end in range(T - 1, len(g), stride):
            X.append(A[end - T + 1 : end + 1])
            y.append(s[end])
            c.append(cell)
    return np.stack(X), np.asarray(y, np.float32), np.asarray(c)


class Scaler:
    def fit(self, X):
        flat = X.reshape(-1, X.shape[-1])
        self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-8
        return self

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)


# ───────────────────────────── metrikler ─────────────────────────────
def per_cell(y, p, cells):
    out = {}
    for c in np.unique(cells):
        m = cells == c
        e = p[m] - y[m]
        out[c] = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e**2).mean())), "n": int(m.sum())}
    return out


def macro_mae(y, p, cells):
    return float(np.mean([v["mae"] for v in per_cell(y, p, cells).values()]))


# ───────────────────────────── modeller ─────────────────────────────
def fit_predict_sklearn(name, Xtr, ytr, Xva, yva, cva, Xte, seed):
    f = lambda X: X.reshape(len(X), -1)
    if name == "mean":
        return np.full(len(Xte), ytr.mean()), {"choice": "train-mean"}
    if name == "last":  # yalnızca son döngünün özellikleri, Ridge
        f = lambda X: X[:, -1, :]
        cands = [(a, Ridge(alpha=a)) for a in (0.1, 1, 10, 100)]
    elif name == "ridge":
        cands = [(a, Ridge(alpha=a)) for a in (0.1, 1, 10, 100)]
    elif name == "hgb":
        cands = [
            (it, HistGradientBoostingRegressor(max_iter=it, learning_rate=0.05, max_leaf_nodes=15, random_state=seed))
            for it in (100, 300)
        ]
    best = None
    for k, m in cands:
        m.fit(f(Xtr), ytr)
        s = macro_mae(yva, m.predict(f(Xva)), cva)
        if best is None or s < best[0]:
            best = (s, k, m)
    return best[2].predict(f(Xte)), {"choice": str(best[1]), "val_macro_mae": best[0]}


class LSTMReg(torch.nn.Module):
    def __init__(self, nf, h=64):
        super().__init__()
        self.lstm = torch.nn.LSTM(nf, h, num_layers=2, batch_first=True, dropout=0.1)
        self.head = torch.nn.Linear(h, 1)

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1]).squeeze(-1)


class HATWrap(torch.nn.Module):
    def __init__(self, nf, T):
        super().__init__()
        from model import HATNet

        self.m = HATNet(n_features=nf, seq_len=T)

    def forward(self, x):
        return self.m(x)[0]


def fit_predict_torch(name, Xtr, ytr, Xva, yva, cva, Xte, seed, epochs, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    nf, T = Xtr.shape[-1], Xtr.shape[1]
    model = (LSTMReg(nf) if name == "lstm" else HATWrap(nf, T)).to(device)
    ymu, ysd = ytr.mean(), ytr.std() + 1e-8
    # HAT-Net SOH kafası sigmoid → hedefi [0,1]'e min-max ile taşı; diğerleri z-skor
    if name == "hat_net":
        lo, hi = ytr.min() - 0.05, ytr.max() + 0.05
        enc = lambda y: (y - lo) / (hi - lo)
        dec = lambda z: z * (hi - lo) + lo
    else:
        enc = lambda y: (y - ymu) / ysd
        dec = lambda z: z * ysd + ymu
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(enc(ytr).astype(np.float32)))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, generator=g)

    def predict(X):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                out.append(model(torch.from_numpy(X[i : i + 2048]).to(device)).cpu().numpy())
        return dec(np.concatenate(out))

    best, patience, wait = (np.inf, 0, None), 8, 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
        s = macro_mae(yva, predict(Xva), cva)
        if s < best[0]:
            best, wait = (s, ep, copy.deepcopy(model.state_dict())), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best[2])
    return predict(Xte), {"choice": f"epoch={best[1]}", "val_macro_mae": best[0]}


# ───────────────────────────── ana döngü ─────────────────────────────
def run(features, models, seeds, folds, T, stride, epochs):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = []
    for fs in features:
        df, cols = load_frame(fs)
        cells = np.array(sorted(df.cell.unique()))
        print(f"\n=== özellik kümesi: {fs} | {len(cells)} hücre, {len(cols)} özellik, T={T}")
        for seed in seeds:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(cells)
            for k, (tr_idx, te_idx) in enumerate(GroupKFold(folds).split(perm, groups=perm)):
                tr_cells, te_cells = perm[tr_idx], perm[te_idx]
                n_val = max(1, len(tr_cells) // 5)
                va_cells, tr_cells = tr_cells[:n_val], tr_cells[n_val:]
                Xtr, ytr, _ = make_windows(df, cols, tr_cells, T, stride)
                Xva, yva, cva = make_windows(df, cols, va_cells, T, stride)
                Xte, yte, cte = make_windows(df, cols, te_cells, T, stride)
                sc = Scaler().fit(Xtr)
                Xtr, Xva, Xte = sc(Xtr), sc(Xva), sc(Xte)
                for mname in models:
                    t0 = time.time()
                    if mname in ("lstm", "hat_net"):
                        p, info = fit_predict_torch(mname, Xtr, ytr, Xva, yva, cva, Xte, seed, epochs, device)
                    else:
                        p, info = fit_predict_sklearn(mname, Xtr, ytr, Xva, yva, cva, Xte, seed)
                    pc = per_cell(yte, p, cte)
                    for c, v in pc.items():
                        records.append({"features": fs, "model": mname, "seed": seed, "fold": k, "cell": c, **v})
                    print(
                        f"  seed={seed} fold={k} {mname:8s} macroMAE={np.mean([v['mae'] for v in pc.values()]):.4f}"
                        f"  ({info['choice']}, {time.time() - t0:.0f}s)",
                        flush=True,
                    )
                pd.DataFrame(records).to_csv(RESULTS / "per_cell_results.csv", index=False)
    return pd.DataFrame(records)


def summarize(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    # hücre başına MAE: tohumlar üzerinde ortala → her hücre bir gözlem
    cell_mae = res.groupby(["features", "model", "cell"]).mae.mean().reset_index()
    for (fs, m), g in cell_mae.groupby(["features", "model"]):
        v = g.mae.to_numpy()
        boots = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        rows.append({"features": fs, "model": m, "n_cells": len(v), "macro_MAE": v.mean(),
                     "CI95_lo": np.percentile(boots, 2.5), "CI95_hi": np.percentile(boots, 97.5),
                     "median_cell_MAE": np.median(v)})
    summ = pd.DataFrame(rows).sort_values(["features", "macro_MAE"])

    tests = []
    for fs, g in cell_mae.groupby("features"):
        piv = g.pivot(index="cell", columns="model", values="mae")
        best = piv.mean().idxmin()
        for m in piv.columns:
            if m == best:
                continue
            d = piv[[best, m]].dropna()
            p = wilcoxon(d[best], d[m]).pvalue if len(d) > 5 else np.nan
            tests.append({"features": fs, "best": best, "vs": m, "n_cells": len(d),
                          "best_wins_cells": int((d[best] < d[m]).sum()), "wilcoxon_p": p})
    return summ, pd.DataFrame(tests)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", default=["ops", "cycle", "curve@3.2", "curve@3.1", "curve@3.0", "curve@2.0", "leaky"])
    ap.add_argument("--models", nargs="+", default=["mean", "last", "ridge", "hgb", "lstm", "hat_net"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    if a.summarize_only:
        res = pd.read_csv(RESULTS / "per_cell_results.csv")
    else:
        res = run(a.features, a.models, a.seeds, a.folds, a.T, a.stride, a.epochs)
    summ, tests = summarize(res)
    summ.to_csv(RESULTS / "summary.csv", index=False)
    tests.to_csv(RESULTS / "wilcoxon.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n", summ.round(4).to_string(index=False))
    print("\n", tests.round(4).to_string(index=False))
    json.dump(vars(a), open(RESULTS / "run_config.json", "w"), indent=1)
