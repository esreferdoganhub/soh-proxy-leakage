"""
scripts/pinn4soh_audit.py
PINN4SOH (Wang ve ark., Nat Commun 2024) ön-işleme tercihlerinin SOH doğruluğuna etkisi;
yazarların dört veri seti (XJTU, TJU, HUST, MIT) ve kendi hücre bölmeleri üzerinde.

Bölmeler main_*.py ile aynıdır. Tek fark: TJU'da yazarlar test hücrelerini os.listdir
sırasından seçer (işletim sistemine bağlı); burada dosyalar ALFABETİK sıralanır.
Değiştirilen tek şey ön-işlemedir:

  A  orijinal     : her hücre KENDİ tüm-ömür min/max'ı ile [-1,1]'e (test hücreleri dahil),
                    'cycle index' özellik, hücre-içi 3-sigma ayıklama
  B  döngüsüz     : A, ama 'cycle index' çıkarılmış
  C  yalnız döngü : A'daki normalize 'cycle index' TEK özellik (≈ ömür oranı)
  D  sızıntısız   : normalizasyon yalnızca EĞİTİM hücrelerinden (global), 'cycle index' yok
  E  D − CC/CV Q  : D, ayrıca şarj segmenti kapasiteleri (CC Q, CV Q) çıkarılmış
  F  sızıntısız+t : D, ama 'cycle index' (PINN'in zaman değişkeni t) korunur ve o da
                    yalnızca eğitim hücrelerinden global ölçeklenir (mutlak döngü sayısı)

Modeller: eğitim ortalaması, Ridge, HGB. Metrik: hücre başına MAE'nin ortalaması
(yazarların 'battery_mean' tablolarıyla aynı), ayrıca havuzlanmış MAE ve RMSE.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "external" / "PINN4SOH" / "data"
XJTU_BATCHES = ["2C", "3C", "R2.5", "R3", "RW", "satellite"]
TJU_BATCHES = ["Dataset_1_NCA_battery", "Dataset_2_NCM_battery", "Dataset_3_NCM_NCA_battery"]
HUST_TEST = {"1-4", "1-8", "2-4", "2-8", "3-4", "3-8", "4-4", "4-8", "5-4", "5-7",
             "6-4", "6-8", "7-4", "7-8", "8-4", "8-8", "9-4", "9-8", "10-4", "10-8"}


def experiments() -> list[dict]:
    """Yazarların main_*.py dosyalarındaki eğitim/test bölmeleri."""
    exps = []
    root = DATA / "XJTU data"
    for b in XJTU_BATCHES:
        files = sorted(f for f in os.listdir(root) if b in f)
        test = [f for f in files if "4" in f or "8" in f]
        exps.append({"dataset": "XJTU", "group": b, "nominal": 2.0, "main": "main_XJTU",
                     "train": [str(root / f) for f in files if f not in test], "test": [str(root / f) for f in test]})
    mod = [(5, 9), (4, 8), (5, 9)]
    nominal = [3.5, 3.5, 2.5]
    for i, b in enumerate(TJU_BATCHES):
        broot = DATA / "TJU data" / b
        files = sorted(os.listdir(broot))
        test = [f for j, f in enumerate(files) if (j + 1) % 10 in mod[i]]
        exps.append({"dataset": "TJU", "group": b.replace("Dataset_", "").replace("_battery", ""), "nominal": nominal[i], "main": "main_TJU",
                     "train": [str(broot / f) for f in files if f not in test], "test": [str(broot / f) for f in test]})
    root = DATA / "HUST data"
    files = sorted(os.listdir(root))
    exps.append({"dataset": "HUST", "group": "all", "nominal": 1.1, "main": "main_HUST",
                 "train": [str(root / f) for f in files if f[:-4] not in HUST_TEST],
                 "test": [str(root / f) for f in files if f[:-4] in HUST_TEST]})
    tr, te = [], []
    for b in ["2017-05-12", "2017-06-30", "2018-04-12"]:
        broot = DATA / "MIT data" / b
        for f in sorted(os.listdir(broot)):
            (te if int(f.split("-")[-1].split(".")[0]) % 5 == 0 else tr).append(str(broot / f))
    exps.append({"dataset": "MIT", "group": "all", "nominal": 1.1, "main": "main_MIT", "train": tr, "test": te})
    return exps


def three_sigma_drop(df):
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    bad = set()
    for c in df.columns:
        s = df[c]
        bad |= set(np.where((s < s.mean() - 3 * s.std()) | (s > s.mean() + 3 * s.std()))[0])
    return df.drop(sorted(bad)).reset_index(drop=True)


def read_cell(path, nominal):
    """Yazarların read_one_csv'sinin normalizasyon ÖNCESİ hali."""
    df = pd.read_csv(path)
    df.insert(df.shape[1] - 1, "cycle index", np.arange(df.shape[0]))
    df = three_sigma_drop(df)
    df["capacity"] = df["capacity"] / nominal
    return df


def per_cell_minmax(df, feats):
    f = df[feats]
    return 2 * (f - f.min()) / (f.max() - f.min()) - 1


def featurize(condition, exp):
    """Hücre başına (X, y) sözlüğü; hücre anahtarı = dosya yolu."""
    cells = {f: read_cell(f, exp["nominal"]) for f in exp["train"] + exp["test"]}
    all_feats = [c for c in next(iter(cells.values())).columns if c != "capacity"]
    if condition in ("A", "B", "C"):
        feats = {"A": all_feats, "B": [c for c in all_feats if c != "cycle index"], "C": ["cycle index"]}[condition]
        X = {f: per_cell_minmax(d, all_feats)[feats].to_numpy() for f, d in cells.items()}
    else:
        drop = (set() if condition == "F" else {"cycle index"}) | ({"CC Q", "CV Q"} if condition == "E" else set())
        feats = [c for c in all_feats if c not in drop]
        tr = pd.concat([cells[f][feats] for f in exp["train"]])
        lo, hi = tr.min(), tr.max()
        X = {f: (2 * (d[feats] - lo) / (hi - lo + 1e-12) - 1).to_numpy() for f, d in cells.items()}
    # Sabit sütun (max==min) → 0/0 = NaN; yazarlarda da aynı olur, 0 ile doldur
    X = {f: np.nan_to_num(x.astype(np.float32)) for f, x in X.items()}
    return {f: (X[f], cells[f]["capacity"].to_numpy(np.float32)) for f in cells}, feats


def stack(arrs, files):
    X = np.vstack([arrs[f][0] for f in files])
    y = np.concatenate([arrs[f][1] for f in files])
    c = np.concatenate([[f] * len(arrs[f][1]) for f in files])
    return X, y, c


def cell_mae(y, p, c):
    return float(np.mean([np.abs(p[c == k] - y[c == k]).mean() for k in np.unique(c)]))


def models():
    return {
        "mean": None,
        "ridge": Ridge(alpha=1.0),
        "hgb": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0),
    }


def main():
    rows = []
    for exp in experiments():
        print(f"{exp['dataset']}/{exp['group']}: {len(exp['train'])} eğitim, {len(exp['test'])} test hücresi", flush=True)
        for cond in "ABCDEF":
            arrs, _ = featurize(cond, exp)
            Xtr, ytr, _ = stack(arrs, exp["train"])
            Xte, yte, cte = stack(arrs, exp["test"])
            for name, m in models().items():
                p = np.full(len(yte), ytr.mean()) if m is None else m.fit(Xtr, ytr).predict(Xte)
                e = p - yte
                rows.append({"dataset": exp["dataset"], "group": exp["group"], "cond": cond, "model": name,
                             "MAE": cell_mae(yte, p, cte), "MAE_pooled": float(np.abs(e).mean()),
                             "RMSE": float(np.sqrt((e ** 2).mean())),
                             "n_train_cells": len(exp["train"]), "n_test_cells": len(exp["test"])})
    r = pd.DataFrame(rows)
    out = ROOT / "results" / "pinn4soh_audit"
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "preprocessing_audit.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.pivot_table(index=["dataset", "group"], columns=["cond", "model"], values="MAE")
          .xs("hgb", axis=1, level=1).mul(100).round(3))
    print(r.pivot_table(index="dataset", columns=["cond", "model"], values="MAE").mul(100).round(3).T)


if __name__ == "__main__":
    sys.exit(main())
