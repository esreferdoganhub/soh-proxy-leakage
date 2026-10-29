"""NASA/CALCE üzerinde literatürdeki vekil özelliklerin testi (plan: results/nasa_calce_replication/PLAN.md).

Özellik basamakları x bölme (S1 hücre içi kronolojik, S2 bir-hücre-dışarıda) x normalizasyon (N1 yalnız eğitim, N2 hücre başına tüm ömür).
Modeller: eğitim ortalaması, ridge (iç doğrulamayla alfa), HGB (denetimdeki sabit ayar + 4 duyarlılık ayarı).
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
OUT = ROOT / "results" / "nasa_calce_replication"
F0 = {"NASA": ["chg_cc_time_win", "chg_T_mean_win", "chg_T_max_win"], "CALCE": ["chg_cc_time_win"]}
RUNGS = {"F0": [], "F1_full_cc_charge_time": ["chg_cc_time_full"], "F2p_partial_discharge_time": ["dis_time_win"],
         "F2_discharge_time_to_cutoff": ["dis_time_to_cut"], "F3_charge_capacity": ["chg_capacity"]}
ALPHAS = 10.0 ** np.arange(-3, 5)
HGB = {"default": dict(max_iter=300, learning_rate=0.05),
       "s1": dict(max_iter=100, learning_rate=0.1), "s2": dict(max_iter=1000, learning_rate=0.02),
       "s3": dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15), "s4": dict(max_iter=300, learning_rate=0.05, l2_regularization=1.0)}


def load():
    n = pd.read_csv(P / "nasa_cycles.csv"); c = pd.read_csv(P / "calce_cycles.csv")
    d = pd.concat([n, c], ignore_index=True)
    d = d[(d.chg_v_start < 4.0) & (d.chg_cc_time_full > 600)]          # geçerli (tam) şarj kaydı
    d["soh"] = d.capacity / d.nominal * 100
    return d


def norm(tr, te, cols, mode, full):
    if mode == "N1":
        mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
        return ((tr[cols] - mu) / sd).to_numpy(), ((te[cols] - mu) / sd).to_numpy()
    g = full.groupby("cell")[cols]                                     # N2: hücrenin tüm ömrü (test hücresi dahil)
    lo, hi = g.transform("min"), g.transform("max")
    z = 2 * (full[cols] - lo) / (hi - lo).replace(0, 1) - 1
    return z.loc[tr.index].to_numpy(), z.loc[te.index].to_numpy()


def splits(d, scheme):
    cells = sorted(d.cell.unique())
    if scheme == "S2":
        for c in cells:
            yield c, d[d.cell != c], d[d.cell == c]
    else:
        tr, te = [], []
        for c in cells:
            g = d[d.cell == c].sort_values("cycle"); k = int(len(g) * 0.7)
            tr.append(g.iloc[:k]); te.append(g.iloc[k:])
        yield "all", pd.concat(tr), pd.concat(te)


def inner_alpha(tr, cols, scheme, mode, full):
    best, ba = np.inf, 1.0
    for a in ALPHAS:
        errs = []
        for _, itr, iva in (splits(tr, "S2") if scheme == "S2" and tr.cell.nunique() > 1 else splits(tr, "S1")):
            Xa, Xb = norm(itr, iva, cols, mode, full)
            errs.append(np.abs(Ridge(alpha=a).fit(Xa, itr.soh).predict(Xb) - iva.soh).mean())
        if np.mean(errs) < best:
            best, ba = np.mean(errs), a
    return ba


def run():
    d = load(); rows = []
    for ds in ["NASA", "CALCE"]:
        dd = d[d.dataset == ds]
        for rung, extra in RUNGS.items():
            cols = F0[ds] + extra
            x = dd.dropna(subset=sum([F0[ds] + e for e in RUNGS.values()], []))   # her basamakta aynı satırlar
            for scheme in ["S1", "S2"]:
                for mode in ["N1", "N2"]:
                    for fold, tr, te in splits(x, scheme):
                        Xtr, Xte = norm(tr, te, cols, mode, x)
                        preds = {"mean": np.full(len(te), tr.soh.mean())}
                        a = inner_alpha(tr, cols, scheme, mode, x)
                        preds["ridge"] = Ridge(alpha=a).fit(Xtr, tr.soh).predict(Xte)
                        for h, kw in HGB.items():
                            preds[f"hgb_{h}"] = HistGradientBoostingRegressor(random_state=0, **kw).fit(Xtr, tr.soh).predict(Xte)
                        for m, p in preds.items():
                            e = np.abs(p - te.soh.to_numpy())
                            for c in te.cell.unique():
                                k = (te.cell == c).to_numpy()
                                rows.append(dict(dataset=ds, rung=rung, scheme=scheme, norm=mode, model=m, cell=c,
                                                 n_test=int(k.sum()), mae=e[k].mean()))
    r = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "per_cell.csv", index=False)
    s = r.groupby(["dataset", "scheme", "norm", "model", "rung"]).mae.mean().unstack("rung")[list(RUNGS)]
    s.to_csv(OUT / "summary.csv")
    cov = d.assign(cov=d.dis_q_at_win_end / d.capacity).groupby(["dataset", "cell"])["cov"].median()
    cov.to_csv(OUT / "partial_window_coverage.csv")
    n = d.groupby(["dataset", "cell"]).size(); n.to_csv(OUT / "n_cycles.csv")
    return s, cov, n


if __name__ == "__main__":
    s, cov, n = run()
    pd.set_option("display.width", 200)
    print(n); print(cov.round(2))
    print(s.xs("N1", level="norm").loc[(slice(None), slice(None), ["mean", "ridge", "hgb_default"]), :].round(2))
    print(s.xs("N2", level="norm").xs("S2", level="scheme").loc[(slice(None), ["ridge", "hgb_default"]), :].round(2))
