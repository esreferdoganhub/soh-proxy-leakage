"""NASA PCoE (B0005/6/7/18) ve CALCE CS2 (35-38) ham verisinden döngü başına özellik tablosu.

Her satır bir deşarj döngüsüdür; özellikler hemen önceki şarjdan ve aynı deşarjdan çıkarılır.
Çıktı: data/processed/nasa_calce_cycles.csv
"""
import glob, os, re, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
NASA = Path(os.environ.get("NASA_DIR", ROOT / "data" / "raw" / "NASA_PCoE" / "5. Battery Data Set" / "FY08Q4"))
CALCE = Path(os.environ.get("CALCE_DIR", ROOT / "data" / "raw" / "CALCE_CS2"))
OUT = ROOT / "data" / "processed" / "nasa_calce_cycles.csv"  # yalnızca klasör/ad kökü; çıktılar <ad>_cycles.csv
CHG_WIN, DIS_WIN, V_CUT = (3.9, 4.1), (3.8, 3.5), 2.7


def t_cross_up(t, v, lo, hi):
    """Şarjda voltajın lo'dan hi'ye çıkma süresi (ilk geçişler); yoksa NaN."""
    a, b = np.argmax(v >= lo), np.argmax(v >= hi)
    return t[b] - t[a] if v.max() >= hi and v[0] < lo and b > a else np.nan


def t_cross_down(t, v, hi, lo):
    a, b = np.argmax(v <= hi), np.argmax(v <= lo)
    return t[b] - t[a] if v.min() <= lo and v[0] > hi and b > a else np.nan


def charge_feats(t, v, i, T):
    cc = i > 0.3                      # sabit akım evresi
    out = {"chg_cc_time_win": t_cross_up(t, v, *CHG_WIN)}
    m = (v >= CHG_WIN[0]) & (v <= CHG_WIN[1]) & cc
    out["chg_T_mean_win"] = T[m].mean() if m.any() else np.nan
    out["chg_T_max_win"] = T[m].max() if m.any() else np.nan
    k = np.argmax(v >= 4.195) if v.max() >= 4.195 else None
    out["chg_cc_time_full"] = t[k] - t[np.argmax(cc)] if k is not None and cc.any() else np.nan
    out["chg_v_start"] = v[np.argmax(cc)] if cc.any() else np.nan
    pos = np.clip(i, 0, None)
    out["chg_capacity"] = np.trapezoid(pos, t) / 3600.0
    return out


def discharge_feats(t, v, i):
    on = i < -0.3
    t0 = t[np.argmax(on)]
    k = np.argmax((v <= V_CUT) & on) if ((v <= V_CUT) & on).any() else None
    out = {"dis_time_to_cut": t[k] - t0 if k is not None else np.nan,
           "dis_time_win": t_cross_down(t, v, *DIS_WIN)}
    neg = np.clip(-i, 0, None)
    q = np.concatenate([[0], np.cumsum(np.diff(t) * (neg[1:] + neg[:-1]) / 2)]) / 3600.0
    qa = q[np.argmax(v <= DIS_WIN[1])] if v.min() <= DIS_WIN[1] else np.nan
    out["dis_q_at_win_end"] = qa
    return out


def nasa():
    rows = []
    for cell in ["B0005", "B0006", "B0007", "B0018"]:
        cyc = sio.loadmat(NASA / f"{cell}.mat", simplify_cells=True)[cell]["cycle"]
        last_chg, k = None, 0
        for c in cyc:
            d = c["data"]
            if c["type"] == "charge":
                last_chg = charge_feats(np.asarray(d["Time"], float), np.asarray(d["Voltage_measured"], float),
                                        np.asarray(d["Current_measured"], float), np.asarray(d["Temperature_measured"], float))
            elif c["type"] == "discharge":
                k += 1
                r = {"dataset": "NASA", "cell": cell, "cycle": k, "capacity": float(d["Capacity"]), "nominal": 2.0}
                r.update(discharge_feats(np.asarray(d["Time"], float), np.asarray(d["Voltage_measured"], float),
                                         np.asarray(d["Current_measured"], float)))
                r.update(last_chg or {})
                last_chg = None          # bir şarj yalnızca bir deşarja eşlenir
                rows.append(r)
    return rows


def calce_files(cell):
    def key(p):
        m, dd, yy = map(int, re.search(r"_(\d+)_(\d+)_(\d+)\.xlsx$", p).groups())
        return (yy, m, dd)
    return sorted(glob.glob(str(CALCE / cell / "*.xlsx")), key=key)


def calce():
    rows = []
    for cell in ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]:
        k = 0
        for f in calce_files(cell):
            x = pd.ExcelFile(f)
            sh = [s for s in x.sheet_names if s.startswith("Channel")]
            d = pd.concat([x.parse(s) for s in sh]) if sh else None
            if d is None or d.empty:
                continue
            for _, g in d.groupby("Cycle_Index", sort=True):
                t = g["Test_Time(s)"].to_numpy(float); v = g["Voltage(V)"].to_numpy(float); i = g["Current(A)"].to_numpy(float)
                dis, chg = i < -0.3, i > 0.05
                if not dis.any() or not chg.any():
                    continue
                kd = np.argmax(dis)
                if np.argmax(chg) > kd:          # bu döngüde şarj deşarjdan sonra gelmeli değilse atla
                    pass
                cq = g["Charge_Capacity(Ah)"].to_numpy(float); dq = g["Discharge_Capacity(Ah)"].to_numpy(float)
                cap = dq.max() - dq.min()
                ch_idx = np.where(chg[:kd])[0]
                if len(ch_idx) < 5:
                    continue
                s, e = ch_idx[0], ch_idx[-1] + 1
                r = {"dataset": "CALCE", "cell": cell, "file": Path(f).name, "capacity": cap, "nominal": 1.1,
                     "dis_I_med": float(np.median(i[dis]))}
                r.update(charge_feats(t[s:e], v[s:e], i[s:e], np.full(e - s, np.nan)))
                r["chg_capacity"] = cq[s:e].max() - cq[s:e].min()
                r.update(discharge_feats(t[kd:], v[kd:], i[kd:]))
                k += 1; r["cycle"] = k
                rows.append(r)
    return rows


if __name__ == "__main__":
    which = sys.argv[1:] or ["nasa", "calce"]
    for w in which:                              # her veri seti ayrı dosyaya: data/processed/<ad>_cycles.csv
        df = pd.DataFrame(nasa() if w == "nasa" else calce())
        out = OUT.with_name(f"{w}_cycles.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(df.groupby(["dataset", "cell"]).agg(n=("capacity", "size"), cap0=("capacity", "first"), capN=("capacity", "last")))
