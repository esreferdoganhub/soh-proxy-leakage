"""
scripts/extract_mit_h5.py
Severson/MIT (Nature Energy 2019) batch .mat dosyalarından GERÇEK döngü
özelliklerini h5py ile akış halinde (tüm dosyayı RAM'e almadan) çıkarır.

Eski `convert_severson_mat.py` voltaj/akım/SEI sütunlarını sabit sayılarla
dolduruyordu (3.3, 4.4, 5000 ...); bu betik onun yerine geçer.

Çıktılar (her batch için, PROC_DIR altında):
  mit_<batch>_cycles.csv   : hücre × döngü skaler özetleri (IR, T, şarj süresi, QD ...)
  mit_<batch>_curves.npz   : hücre × döngü × 50 noktalı Qdlin(V) ve Tdlin(V) eğrileri
  mit_<batch>_cells.csv    : hücre meta verisi (politika, barkod, cycle_life)

Hücre birleştirme/eleme kuralları Severson starter kodundaki (rdbraatz GitHub
deposu, "Load Data" not defteri) standart 124 hücre kümesini üretir; bkz. `merge_batches`.

Kullanım:
  python notebooks/extract_mit_h5.py /path/to/2017-05-12_...mat [--delete-after]
  python notebooks/extract_mit_h5.py --merge
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))  # config.PROC_DIR = data/processed
from config import PROC_DIR  # noqa: E402

MIT_URLS = {
    "2017-05-12": "https://data.matr.io/1/api/v1/file/5c86c0b5fa2ede00015ddf66/download",  # 3.03 GB
    "2017-06-30": "https://data.matr.io/1/api/v1/file/5c86bf13fa2ede00015ddd82/download",  # 2.01 GB
    "2018-04-12": "https://data.matr.io/1/api/v1/file/5c86bd64fa2ede00015ddbb2/download",  # 3.24 GB
}
BATCH_TAG = {"2017-05-12": "b1", "2017-06-30": "b2", "2018-04-12": "b3"}

SUMMARY_KEYS = ["cycle", "QDischarge", "QCharge", "IR", "Tmax", "Tavg", "Tmin", "chargetime"]
N_CURVE = 50  # Qdlin 1000 noktalı (3.6 V → 2.0 V); 50 noktaya seyreltilir
CURVE_IDX = np.linspace(0, 999, N_CURVE).round().astype(int)


def _str(f, ref) -> str:
    a = np.asarray(f[ref]).ravel()
    if a.dtype == np.uint16:  # MATLAB char dizisi
        return "".join(chr(c) for c in a)
    return " ".join(str(x) for x in a)  # bazı alanlar (barcode) sayısal saklanmış


def _arr(f, ref) -> np.ndarray:
    return np.asarray(f[ref]).ravel()


def extract_batch(mat_path: Path) -> str:
    date = mat_path.name[:10]
    tag = BATCH_TAG[date]
    rows, curves_q, curves_t, keys, meta = [], [], [], [], []

    with h5py.File(mat_path, "r") as f:
        batch = f["batch"]
        n_cells = batch["summary"].shape[0]
        print(f"⚡ {mat_path.name}: {n_cells} hücre")
        for i in range(n_cells):
            key = f"{tag}c{i}"
            summ = f[batch["summary"][i, 0]]
            s = {k: np.asarray(summ[k]).ravel() for k in SUMMARY_KEYS}
            n = len(s["QDischarge"])

            cyc = f[batch["cycles"][i, 0]]
            qd_refs, td_refs = cyc["Qdlin"], cyc["Tdlin"]
            n_c = min(n, qd_refs.shape[0])
            q = np.full((n, N_CURVE), np.nan, dtype=np.float32)
            t = np.full((n, N_CURVE), np.nan, dtype=np.float32)
            for j in range(n_c):
                qj = _arr(f, qd_refs[j, 0])
                tj = _arr(f, td_refs[j, 0])
                if qj.size == 1000:
                    q[j] = qj[CURVE_IDX]
                if tj.size == 1000:
                    t[j] = tj[CURVE_IDX]

            df = pd.DataFrame({k: s[k][:n] for k in SUMMARY_KEYS})
            df.insert(0, "cell", key)
            df["row"] = np.arange(n)
            rows.append(df)
            curves_q.append(q)
            curves_t.append(t)
            keys.append(np.full(n, key))

            cl = _arr(f, batch["cycle_life"][i, 0])
            meta.append({
                "cell": key,
                "policy": _str(f, batch["policy_readable"][i, 0]),
                "barcode": _str(f, batch["barcode"][i, 0]),
                "channel": _str(f, batch["channel_id"][i, 0]),
                "cycle_life_reported": float(cl[0]) if cl.size else np.nan,
                "n_cycles": n,
            })
            print(f"  {key}: {n} döngü, QD son={s['QDischarge'][-1]:.3f}")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(PROC_DIR / f"mit_{tag}_cycles.csv", index=False)
    pd.DataFrame(meta).to_csv(PROC_DIR / f"mit_{tag}_cells.csv", index=False)
    np.savez_compressed(
        PROC_DIR / f"mit_{tag}_curves.npz",
        cell=np.concatenate(keys), qdlin=np.vstack(curves_q), tdlin=np.vstack(curves_t),
        voltage=np.linspace(3.6, 2.0, 1000)[CURVE_IDX],
    )
    print(f"💾 {tag} kaydedildi → {PROC_DIR}")
    return tag


# Severson starter kodu: b2'deki 5 hücre b1c0–b1c4'ün devamıdır; bazı hücreler elenir.
B1_REMOVE = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]
B2_CONTINUES_B1 = {"b2c7": "b1c0", "b2c8": "b1c1", "b2c9": "b1c2", "b2c15": "b1c3", "b2c16": "b1c4"}
B3_REMOVE = ["b3c37", "b3c2", "b3c23", "b3c32", "b3c42", "b3c43"]


def merge_batches() -> None:
    cyc = pd.concat([pd.read_csv(PROC_DIR / f"mit_{t}_cycles.csv") for t in ("b1", "b2", "b3")])
    meta = pd.concat([pd.read_csv(PROC_DIR / f"mit_{t}_cells.csv") for t in ("b1", "b2", "b3")])
    cur = [np.load(PROC_DIR / f"mit_{t}_curves.npz") for t in ("b1", "b2", "b3")]
    q = np.vstack([c["qdlin"] for c in cur])
    tq = np.vstack([c["tdlin"] for c in cur])
    cyc = cyc.reset_index(drop=True)
    cyc["curve_row"] = np.arange(len(cyc))

    # b2 devam hücrelerini b1 hücresine ekle (döngü indeksini kaydırarak)
    for b2, b1 in B2_CONTINUES_B1.items():
        last = cyc.loc[cyc.cell == b1, "row"].max()
        m = cyc.cell == b2
        cyc.loc[m, "row"] = cyc.loc[m, "row"] + last + 1
        cyc.loc[m, "cell"] = b1
    drop = set(B1_REMOVE) | set(B3_REMOVE)
    cyc = cyc[~cyc.cell.isin(drop)].sort_values(["cell", "row"]).reset_index(drop=True)
    meta = meta[~meta.cell.isin(drop | set(B2_CONTINUES_B1))]

    out = PROC_DIR / "mit_all_cycles.csv"
    cyc.to_csv(out, index=False)
    meta.to_csv(PROC_DIR / "mit_all_cells.csv", index=False)
    np.savez_compressed(PROC_DIR / "mit_all_curves.npz", qdlin=q, tdlin=tq, voltage=cur[0]["voltage"])
    print(f"✅ Birleştirildi: {cyc.cell.nunique()} hücre, {len(cyc):,} döngü → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mat", nargs="?", type=Path)
    ap.add_argument("--delete-after", action="store_true")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.mat:
        extract_batch(a.mat)
        if a.delete_after:
            a.mat.unlink()
            print(f"🗑️  {a.mat.name} silindi")
    if a.merge:
        merge_batches()
