"""Ek analizler (hakem yanıtları): üç sonuç dosyasını yeniden üretir.

  results/mit_tuned_k50/k8_vs_k50.csv              K=8 ve K=50 ayar bütçesi karşılaştırması
  results/pinn4soh_audit/hgb_sensitivity.csv       PINN denetiminde HGB'nin 4 ayara duyarlılığı
  results/pinn4soh_audit/lifetime_cv_hypothesis.csv ömür CV'si hipotezi (keşifsel, sonradan)

Kullanım: python scripts/supplementary_analyses.py [k50] [hgb] [lifecv]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr, pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "scripts"))


def k8_vs_k50():
    d = pd.read_csv(R / "mit_tuned_k50" / "final_per_cell.csv"); o = pd.read_csv(R / "mit_tuned" / "final_per_cell.csv")
    s = pd.read_csv(R / "mit_tuned_k50" / "summary.csv")
    f = lambda x, fs, m: x[(x.features == fs) & (x.model == m)].groupby("cell").mae.mean()
    rows = []
    for fs in ["ops", "curve@3.2"]:
        for m in ["hgb", "lstm", "transformer"]:
            r = s[(s.features == fs) & (s.model == m)].iloc[0]
            rows.append(dict(set=fs, model=m, K8=f(o, fs, m).mean() * 100, K50=r.macro_MAE * 100, lo=r.CI95_lo * 100, hi=r.CI95_hi * 100))
    t = pd.DataFrame(rows); print(t.round(3).to_string(index=False))
    for fs in ["ops", "curve@3.2"]:
        h = f(d, fs, "hgb")
        for m in ["lstm", "transformer"]:
            x = f(d, fs, m); print(f"{fs:10s} K50 HGB vs {m:11s}: HGB better {int((h < x).sum())}/124, p={wilcoxon(h, x).pvalue:.2g}")
    t.to_csv(R / "mit_tuned_k50" / "k8_vs_k50.csv", index=False)


def hgb_sensitivity():
    import pinn4soh_audit as au
    p = pd.read_csv(R / "pinn4soh_audit" / "pinn_results.csv").groupby(["dataset", "group", "cond"]).MAE.mean()
    cfgs = {"default(300,0.05)": dict(max_iter=300, learning_rate=0.05), "100,0.1": dict(max_iter=100, learning_rate=0.1),
            "1000,0.02": dict(max_iter=1000, learning_rate=0.02), "300,0.05,leaves15": dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15)}
    rows = []
    for e in au.experiments():
        for cond in "AF":
            arrs, _ = au.featurize(cond, e); Xtr, ytr, _ = au.stack(arrs, e["train"]); Xte, yte, cte = au.stack(arrs, e["test"])
            for name, c in cfgs.items():
                m = HistGradientBoostingRegressor(random_state=0, **c).fit(Xtr, ytr)
                rows.append(dict(exp=f"{e['dataset']}/{e['group']}", cond=cond, cfg=name, mae=au.cell_mae(yte, m.predict(Xte), cte),
                                 pinn=p[(e["dataset"], e["group"], cond)]))
    d = pd.DataFrame(rows)
    for (cond, cfg), g in d.groupby(["cond", "cfg"]):
        print(f"{cond} {cfg:18s} HGB={g.mae.mean()*100:.3f} PINN={g.pinn.mean()*100:.3f} wins {int((g.mae < g.pinn).sum())}/11 p={wilcoxon(g.mae, g.pinn).pvalue:.3f}")
    d.to_csv(R / "pinn4soh_audit" / "hgb_sensitivity.csv", index=False)


def lifetime_cv():
    import pinn4soh_audit as au
    a = pd.read_csv(R / "pinn4soh_audit" / "preprocessing_audit.csv")
    p = pd.read_csv(R / "pinn4soh_audit" / "pinn_results.csv")
    rows = []
    for e in au.experiments():
        L = np.array([len(au.read_cell(f, e["nominal"])) for f in e["train"] + e["test"]])  # yazarların ön-işlemesi (3-sigma) sonrası döngü sayısı
        g = a[(a.dataset == e["dataset"]) & (a.group == e["group"])]
        m = lambda c, mod: g[(g.cond == c) & (g.model == mod)].MAE.iloc[0]
        pp = p[(p.dataset == e["dataset"]) & (p.group == e["group"])].groupby("cond").MAE.mean()
        rows.append({"exp": f"{e['dataset']}/{e['group']}", "n_cells": len(L), "life_mean": L.mean(), "life_cv": L.std() / L.mean(),
                     "skill_C": 1 - m("C", "hgb") / m("C", "mean"),
                     "pinn_norm_gain": (pp["F"] - pp["A"]) * 100, "hgb_norm_gain": (m("F", "hgb") - m("A", "hgb")) * 100})
    d = pd.DataFrame(rows).sort_values("life_cv")
    for y in ["skill_C", "pinn_norm_gain", "hgb_norm_gain"]:
        r, pv = spearmanr(d.life_cv, d[y]); rp, pp_ = pearsonr(d.life_cv, d[y])
        print(f"{y:15s} vs life_cv: Spearman rho={r:+.2f} (p={pv:.3f}) | Pearson r={rp:+.2f} (p={pp_:.3f}) n={len(d)}")
    d.to_csv(R / "pinn4soh_audit" / "lifetime_cv_hypothesis.csv", index=False)


if __name__ == "__main__":
    which = sys.argv[1:] or ["k50", "hgb", "lifecv"]
    if "k50" in which: k8_vs_k50()
    if "hgb" in which: hgb_sensitivity()
    if "lifecv" in which: lifetime_cv()
