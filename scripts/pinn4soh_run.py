"""
scripts/pinn4soh_run.py
Yazarların PINN kodunu (external/PINN4SOH, değiştirilmeden) iki ön-işleme koşulunda çalıştırır:
  A : orijinal (hücre-başına tüm-ömür min-max; t = ömre göre normalize döngü)
  F : sızıntısız (yalnızca eğitim hücrelerinden global min-max; t = mutlak döngü, global ölçek)
Bölmeler, eğitim/doğrulama ayrımı, hiperparametreler (her veri setinin main_*.py
varsayılanları) ve model seçimi yazarlarınkiyle aynıdır.

Kullanım: python scripts/pinn4soh_run.py --datasets XJTU TJU HUST MIT --conds A F --repeats 10
"""

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "external" / "PINN4SOH"
sys.path.insert(0, str(EXT))
sys.path.insert(0, str(ROOT / "scripts"))
from Model.Model import PINN  # noqa: E402
import pinn4soh_audit as au  # noqa: E402

OUT = ROOT / "results" / "pinn4soh_audit" / "pinn_runs"


def pairs(arrs, files):
    X1, X2, Y1, Y2 = [], [], [], []
    for f in files:
        x, y = arrs[f]
        X1.append(x[:-1]); X2.append(x[1:]); Y1.append(y[:-1]); Y2.append(y[1:])
    t = lambda a: torch.from_numpy(np.concatenate(a)).float()
    return t(X1), t(X2), t(Y1).view(-1, 1), t(Y2).view(-1, 1)


def loaders(arrs, exp, bs):
    X1, X2, Y1, Y2 = pairs(arrs, exp["train"])
    tr = train_test_split(X1, X2, Y1, Y2, test_size=0.2, random_state=420)  # yazarlarla aynı
    train = DataLoader(TensorDataset(tr[0], tr[2], tr[4], tr[6]), batch_size=bs, shuffle=True)
    valid = DataLoader(TensorDataset(tr[1], tr[3], tr[5], tr[7]), batch_size=bs, shuffle=True)
    cells = np.concatenate([[f] * (len(arrs[f][1]) - 1) for f in exp["test"]])
    test = DataLoader(TensorDataset(*pairs(arrs, exp["test"])), batch_size=bs, shuffle=False)
    return train, valid, test, cells


def authors_args(main_name):
    """İlgili main_*.py'nin get_args() varsayılanları."""
    saved_argv, sys.argv = sys.argv, sys.argv[:1]
    os.chdir(EXT)
    args = importlib.import_module(main_name).get_args()
    sys.argv = saved_argv
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["XJTU", "TJU", "HUST", "MIT"])
    ap.add_argument("--conds", nargs="+", default=["A", "F"])
    ap.add_argument("--repeats", type=int, default=10)
    cli = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = OUT.parent / "pinn_results.csv"  # her koşunun klasörü OUT altında (git dışı)
    done = set()
    if ledger.exists():
        d = pd.read_csv(ledger)
        done = set(zip(d.dataset, d.group, d.cond, d.repeat))
    torch.set_num_threads(2)
    for exp in au.experiments():
        if exp["dataset"] not in cli.datasets:
            continue
        args = authors_args(exp["main"])
        for cond in cli.conds:
            arrs, feats = au.featurize(cond, exp)
            assert feats[-1] == "cycle index"  # PINN: son sütun = t
            for r in range(cli.repeats):
                if (exp["dataset"], exp["group"], cond, r) in done:
                    continue
                torch.manual_seed(r)
                np.random.seed(r)
                folder = OUT / f"{exp['dataset']}_{exp['group']}_{cond}_r{r}"
                args.save_folder, args.log_dir = str(folder), "logging.txt"
                tr, va, te, cells = loaders(arrs, exp, args.batch_size)
                pinn = PINN(args)
                pinn.Train(trainloader=tr, validloader=va, testloader=te)
                y = np.load(folder / "true_label.npy").ravel()
                p = np.load(folder / "pred_label.npy").ravel()
                row = {"dataset": exp["dataset"], "group": exp["group"], "cond": cond, "repeat": r,
                       "MAE": au.cell_mae(y, p, cells), "RMSE": float(np.sqrt(((p - y) ** 2).mean()))}
                pd.DataFrame([row]).to_csv(ledger, mode="a", header=not ledger.exists(), index=False)
                print(f"{exp['dataset']}/{exp['group']} {cond} r{r}  MAE={row['MAE'] * 100:.3f}%", flush=True)
    d = pd.read_csv(ledger)
    print(d.pivot_table(index=["dataset", "group"], columns="cond", values="MAE").mul(100).round(3))


if __name__ == "__main__":
    main()
