"""
scripts/make_figures_paper1.py
Makale 1 (vekil sızıntısı + PINN denetimi) şekilleri — statik, dergi formatı (PDF + 300 dpi PNG).

Renkler (dataviz referans paleti, doğrulandı — adjacent CVD ΔE ≥ 9.1, normal ≥ 19.6;
ilk üç slot all-pairs geçer). Her model TÜM şekillerde aynı renk + aynı işaret:
  HGB=mavi ●, LSTM=turuncu ■, Transformer=su yeşili ▲, Ridge=sarı ◆, HAT-Net=pembe ▼, PINN=mor.
Düşük kontrastlı renkler (sarı, pembe, su yeşili) için ikincil kodlama: işaret şekli + doğrudan etiket.

Girdiler : results/mit_tuned/{summary,final_per_cell}.csv, results/pinn4soh_audit/*
Çıktı    : results/figures_paper1/fig{1..5}_*.{pdf,png}
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL = {  # renk, işaret, etiket
    "hgb": ("#2a78d6", "o", "Gradient boosting (HGB)"),
    "lstm": ("#eb6834", "s", "LSTM"),
    "transformer": ("#1baf7a", "^", "Transformer"),
    "ridge": ("#eda100", "D", "Ridge"),
    "hat_net": ("#e87ba4", "v", "HAT-Net (CNN-Transformer)"),
    "pinn": ("#4a3aa7", "P", "PINN (Wang et al., 2024)"),
}
FEATS = ["ops", "cycle", "curve@3.2", "curve@3.1", "curve@3.0", "curve@2.0", "leaky"]
FEAT_LABEL = {
    "ops": "BMS signals only\n(IR, T, charge time)",
    "cycle": "+ cycle count",
    "curve@3.2": "+ discharge curve ≥3.2 V\n(~21 % of capacity)",
    "curve@3.1": "+ curve ≥3.1 V (~58 %)",
    "curve@3.0": "+ curve ≥3.0 V (~74 %)",
    "curve@2.0": "+ full discharge curve",
    "leaky": "+ charge capacity\n(QCharge, proxy)",
}

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def load_summary():
    s = pd.read_csv(R / "mit_tuned" / "summary.csv")
    for c in ["macro_MAE", "CI95_lo", "CI95_hi", "median_cell_MAE"]:
        s[c] *= 100  # % SOH
    return s


# ─────────────── Şekil 1: vekil şişirmesi (HGB, tek seri) ───────────────
def fig1(s):
    d = s[s.model == "hgb"].set_index("features").loc[FEATS]
    mean_base = s[(s.model == "mean") & (s.features == "ops")].macro_MAE.iloc[0]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    y = np.arange(len(FEATS))[::-1]
    c = MODEL["hgb"][0]
    ax.hlines(y, d.CI95_lo, d.CI95_hi, color=c, linewidth=2, zorder=3)
    ax.scatter(d.macro_MAE, y, s=34, color=c, edgecolor=SURFACE, linewidth=1.2, zorder=4)
    for yi, (f, r) in zip(y, d.iterrows()):
        ax.text(r.CI95_hi * 1.12, yi, f"{r.macro_MAE:.2f}", va="center", fontsize=8, color=INK)
    ax.axvline(mean_base, color=INK2, linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(mean_base * 0.95, y[-1] - 0.1, f"predicting the\ntraining mean\n({mean_base:.2f})", ha="right", va="bottom", fontsize=7.2, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 6)
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
    ax.set_yticks(y)
    ax.set_yticklabels([FEAT_LABEL[f] for f in FEATS], fontsize=7.8)
    ax.set_xlabel("SOH error, cell-averaged MAE (% of nominal, log scale)")
    ratio = d.loc["ops", "macro_MAE"] / d.loc["leaky", "macro_MAE"]
    ax.annotate(f"one proxy feature:\n{ratio:.0f}× lower apparent error", xy=(d.loc["leaky", "macro_MAE"], y[-1] + 0.12),
                xytext=(0.075, y[-1] + 1.35), fontsize=7.5, color=INK, va="center",
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.6))
    ax.set_title("Proxy features inflate apparent SOH accuracy\n(MIT/Severson, 124 cells, gradient boosting, cell-held-out)", loc="left")
    grid(ax)
    save(fig, "fig1_proxy_inflation")


# ─────────────── Şekil 2: model karşılaştırması (sızıntısız kümeler) ───────────────
def fig2(s):
    feats = FEATS[:-1]
    models = ["hgb", "lstm", "transformer", "ridge", "hat_net"]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    base = np.arange(len(feats))[::-1] * 1.0
    off = np.linspace(0.3, -0.3, len(models))
    for m, o in zip(models, off):
        d = s[s.model == m].set_index("features").reindex(feats)
        c, mk, lab = MODEL[m]
        ok = d.macro_MAE.notna()
        yy = base[ok.values] + o
        ax.hlines(yy, d.CI95_lo[ok], d.CI95_hi[ok], color=c, linewidth=1.6, zorder=3)
        ax.scatter(d.macro_MAE[ok], yy, marker=mk, s=26, color=c, edgecolor=SURFACE, linewidth=0.8, zorder=4, label=lab)
    for yb in base[:-1] - 0.5:
        ax.axhline(yb, color=GRID, linewidth=0.6)
    ax.set_yticks(base)
    ax.set_yticklabels([FEAT_LABEL[f] for f in feats], fontsize=7.8)
    ax.set_xlabel("SOH error, cell-averaged MAE (% of nominal) with 95 % bootstrap CI over cells")
    ax.set_xlim(0.6, 2.45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.13), ncol=3, fontsize=7.6, handletextpad=0.3, columnspacing=1.0)
    ax.text(2.43, base[-2] - 0.62, "HAT-Net not run on ≥3.1 V / ≥3.0 V sets (cost)", ha="right", fontsize=6.8, color=INK2)
    grid(ax)
    save(fig, "fig2_model_comparison")


# ─────────────── Şekil 3: kısmi eğri doz-yanıt ───────────────
def fig3(s):
    x = {"ops": 0, "curve@3.2": 21, "curve@3.1": 58, "curve@3.0": 74, "curve@2.0": 100}
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for m in ["hgb", "lstm", "transformer"]:
        c, mk, lab = MODEL[m]
        d = s[(s.model == m) & s.features.isin(x)].copy()
        d["x"] = d.features.map(x)
        d = d.sort_values("x")
        ax.fill_between(d.x, d.CI95_lo, d.CI95_hi, color=c, alpha=0.12, linewidth=0)
        ax.plot(d.x, d.macro_MAE, color=c, linewidth=2, marker=mk, markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=0.8, label=lab)
    h = s[(s.model == "hgb")].set_index("features").macro_MAE
    gain_total = h["ops"] - h["curve@2.0"]
    gain21 = h["ops"] - h["curve@3.2"]
    ax.annotate(f"{gain21 / gain_total * 100:.0f} % of the full-curve gain\nwith 21 % of capacity observed",
                xy=(21, h["curve@3.2"] + 0.03), xytext=(30, 1.55), fontsize=7.3, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.6))
    ax.set_xlabel("Share of capacity covered by the observed\ndischarge-curve window (%)")
    ax.set_ylabel("SOH MAE (% of nominal)")
    ax.set_xticks([0, 21, 58, 74, 100])
    ax.set_ylim(0.7, 2.0)
    ax.legend(fontsize=7.4, loc="upper right")
    grid(ax, "y")
    save(fig, "fig3_partial_curve")


# ─────────────── Şekil 4: PINN denetimi ───────────────
def fig4():
    a = pd.read_csv(R / "pinn4soh_audit" / "preprocessing_audit.csv")
    p = pd.read_csv(R / "pinn4soh_audit" / "pinn_results.csv")
    n = a.groupby(["dataset", "group"]).n_test_cells.first()
    pin = p.groupby(["dataset", "group", "cond"]).MAE.mean().unstack()
    hgb = a[a.model == "hgb"].pivot_table(index=["dataset", "group"], columns="cond", values="MAE")
    order = ["XJTU", "TJU", "HUST", "MIT"]
    w = lambda t: (t.mul(n, axis=0).groupby(level=0).sum().div(n.groupby(level=0).sum(), axis=0) * 100).loc[order]
    P, H = w(pin[["A", "F"]]), w(hgb[["A", "F"]])
    ncell = n.groupby(level=0).sum().loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    titles = {"A": "Authors' preprocessing\n(per-cell whole-life normalisation)",
              "F": "Deployable preprocessing\n(training-set normalisation only)"}
    x = np.arange(len(order))
    bw = 0.36
    for ax, cond in zip(axes, ["A", "F"]):
        for j, (lab, t, key) in enumerate([("PINN", P, "pinn"), ("HGB", H, "hgb")]):
            c = MODEL[key][0]
            xs = x + (j - 0.5) * (bw + 0.02)
            ax.bar(xs, t[cond], width=bw, color=c, zorder=3, label=MODEL[key][2] if cond == "A" else None)
            for xi, v in zip(xs, t[cond]):
                ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.8, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}\n({ncell[d]} cells)" for d in order], fontsize=7.6)
        ax.set_title(titles[cond], fontsize=8.3, loc="left")
        grid(ax, "y")
    axes[0].set_ylabel("SOH MAE (% of nominal)")
    axes[0].legend(fontsize=7.4, loc="upper left", bbox_to_anchor=(0, 1.32), ncol=2)
    save(fig, "fig4_pinn_audit")
    return P, H


# ─────────────── Şekil 5: maliyet–doğruluk ───────────────
def fig5(s):
    d = pd.read_csv(R / "mit_tuned" / "final_per_cell.csv")
    secs = d.drop_duplicates(["features", "model", "seed", "fold"]).groupby("model").secs.median()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    lab_off = {"hgb": (1.3, 0.0), "lstm": (1.3, 0.0), "transformer": (1.3, 0.0), "ridge": (1.6, 0.0), "hat_net": (0.62, 0.0)}
    for m in ["ridge", "hgb", "lstm", "transformer", "hat_net"]:
        c, mk, lab = MODEL[m]
        r = s[(s.model == m) & (s.features == "ops")].iloc[0]
        t = max(secs[m], 0.05)
        ax.vlines(t, r.CI95_lo, r.CI95_hi, color=c, linewidth=1.6, zorder=3)
        ax.scatter(t, r.macro_MAE, marker=mk, s=40, color=c, edgecolor=SURFACE, linewidth=0.8, zorder=4)
        fx, _ = lab_off[m]
        name = lab.split(" (")[0] + (" (<0.1 s)" if secs[m] < 0.05 else "")
        ax.text(t * fx, r.macro_MAE, name, fontsize=7.4, va="center", ha="left" if fx > 1 else "right", color=INK)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 2000)
    ax.set_ylim(1.2, 2.4)
    ax.set_xlabel("Training time per fit on a laptop (s, log scale)")
    ax.set_ylabel("SOH MAE (% of nominal), BMS signals only")
    grid(ax, "y")
    save(fig, "fig5_cost_accuracy")


if __name__ == "__main__":
    s = load_summary()
    fig1(s)
    fig2(s)
    fig3(s)
    P, H = fig4()
    fig5(s)
    print("PINN (A,F):\n", P.round(2), "\nHGB (A,F):\n", H.round(2))
    print("Şekiller:", sorted(p.name for p in OUT.glob("*.png")))
