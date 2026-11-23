"""Generate 6 publication-grade PDF/PNG figures from survival pipeline CSVs."""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTE = {
    "navy": "#1B365D",
    "teal": "#008080",
    "coral": "#D95F02",
    "purple": "#7570B3",
    "gold": "#E6AB02",
    "crimson": "#B2182B",
    "slate": "#4D4D4D",
    "blue": "#2166AC",
    "green": "#1B7837",
}

FEAT_LABELS = [
    "Age", "Male Sex", "Education", "APOE e4", "CDR-SB", "FAQ", "Trail A", "Trail B", "Animals"
]


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.3,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "savefig.dpi": 300,
    })


def _save(fig: plt.Figure, out_dir: str, stem: str) -> None:
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _yr(df: pd.DataFrame) -> pd.Series:
    return df["cutoff_label"].replace("2026_final", "2026").astype(int)


def make_fig1(res_dir: str, fig_dir: str) -> None:
    """Figure 1: Three-Trajectory Causal Decomposition of 36m E/O Calibration (Traj A vs Traj C)."""
    df_a = pd.read_csv(os.path.join(res_dir, "trajectory_a_calendar_cutoff.csv"))
    df_c = pd.read_csv(os.path.join(res_dir, "trajectory_c_locked_unfreezing.csv"))
    df_a["yr"] = _yr(df_a)
    df_c["yr"] = _yr(df_c)

    specs = [
        ("ADNI_S1_mci_to_dem", "A. ADNI: MCI to Dementia (36m E/O)"),
        ("ADNI_S3_cdrsb45", "B. ADNI: CDR-SB >= 4.5 (36m E/O)"),
        ("NACC_S1_mci_to_dem", "C. NACC: MCI to Dementia (36m E/O)"),
        ("NACC_S2_cn_to_imp", "D. NACC: Normal to MCI/Dementia (36m E/O)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8))
    for ax, (ep, title) in zip(axes.flat, specs):
        sa = df_a[(df_a["endpoint_id"] == ep) & (df_a["model_type"] == "cox_std")].sort_values("yr")
        sc = df_c[(df_c["endpoint_id"] == ep) & (df_c["model_type"] == "cox_std")].sort_values("yr")
        sa_r = df_a[(df_a["endpoint_id"] == ep) & (df_a["model_type"] == "cox_ridge_cv")].sort_values("yr")

        ax.axhline(1.0, color="black", lw=1.0, ls=":", alpha=0.75, label="Ideal calibration (E/O = 1.0)")
        ax.plot(sa["yr"], sa["oe_ratio_36m"], color=PALETTE["navy"], lw=2.2, marker="o", ms=4, label="Traj. A: Full calendar evolution (Cox std)")
        ax.plot(sc["yr"], sc["oe_ratio_36m"], color=PALETTE["crimson"], lw=2.2, ls="--", marker="s", ms=4, label="Traj. C: Locked <=2008 unfreezing (pure maturation)")
        ax.plot(sa_r["yr"], sa_r["oe_ratio_36m"], color=PALETTE["teal"], lw=1.6, ls="-.", marker="^", ms=3.5, alpha=0.85, label="Traj. A: Adaptive Ridge-Cox (5-fold CV)")

        # Shade residual case-mix / accrual gap between C(2026) and A(2026)
        c26 = float(sc[sc["cutoff_label"] == "2026_final"]["oe_ratio_36m"].iloc[0])
        a26 = float(sa[sa["cutoff_label"] == "2026_final"]["oe_ratio_36m"].iloc[0])
        if abs(c26 - a26) > 0.12:
            ax.annotate("", xy=(2026.15, a26), xytext=(2026.15, c26),
                        arrowprops=dict(arrowstyle="<->", color=PALETTE["coral"], lw=1.6))
            ax.text(2025.6, 0.5 * (a26 + c26), f"Accrual/\ncase-mix\n({c26 - a26:+.2f})",
                    fontsize=7.4, color=PALETTE["coral"], ha="right", va="center", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.0))

        ax.set_xticks([2008, 2011, 2014, 2017, 2020, 2023, 2026])
        ax.set_xlim(2007.5, 2026.8)
        lo_y, hi_y = ax.get_ylim()
        ax.set_ylim(max(0.75, lo_y - 0.05), hi_y + 0.12)
        ax.set_title(title)
        ax.set_xlabel("Calendar Cutoff Year")
        ax.set_ylabel("36-Month Expected / Observed Ratio (E/O)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, fig_dir, "fig1_three_trajectory_causal_decomposition")


def make_fig2(res_dir: str, fig_dir: str) -> None:
    """Figure 2: Rolling Prospective Survival Transportability (Expanding vs 5-Year Sliding Window)."""
    df_r = pd.read_csv(os.path.join(res_dir, "rolling_prospective.csv"))
    specs = [
        ("ADNI_S1_mci_to_dem", "A. ADNI: MCI to Dementia"),
        ("ADNI_S3_cdrsb45", "B. ADNI: CDR-SB >= 4.5"),
        ("NACC_S1_mci_to_dem", "C. NACC: MCI to Dementia"),
        ("NACC_S2_cn_to_imp", "D. NACC: Normal to MCI/Dementia"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8))
    for ax, (ep, title) in zip(axes.flat, specs):
        sub = df_r[(df_r["endpoint_id"] == ep) & (df_r["model_type"] == "cox_std")]
        s_exp = sub[sub["window_strategy"] == "expanding"].sort_values("deploy_year")
        s_sli = sub[sub["window_strategy"] == "sliding_5y"].sort_values("deploy_year")

        ax.axhline(1.0, color="black", lw=1.0, ls=":", alpha=0.75, label="Ideal 36m E/O = 1.0")
        ax.plot(s_exp["deploy_year"], s_exp["oe_ratio_36m"], color=PALETTE["crimson"], lw=2.2, marker="s", ms=4.5, label="Expanding window (36m E/O)")
        if len(s_sli) > 0:
            ax.plot(s_sli["deploy_year"], s_sli["oe_ratio_36m"], color=PALETTE["teal"], lw=2.0, ls="--", marker="D", ms=4.0, label="5-year sliding window (36m E/O)")

        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.plot(s_exp["deploy_year"], s_exp["c_harrell"], color=PALETTE["slate"], lw=1.5, ls=":", marker="o", ms=3.2, alpha=0.8, label="Expanding Harrell C-index (right)")
        ax2.set_ylabel("Future Cohort Harrell C-index", color=PALETTE["slate"], fontsize=8.5)
        ax2.tick_params(axis="y", labelcolor=PALETTE["slate"], labelsize=8.0)
        ax2.set_ylim(0.60, 0.98)

        ax.set_xticks([2009, 2011, 2013, 2015, 2017, 2019, 2021, 2022])
        ax.set_title(title)
        ax.set_xlabel("Prospective Deployment Cutoff Year")
        ax.set_ylabel("Future Cohort 36m E/O Ratio")

    h1, l1 = axes[0, 0].get_legend_handles_labels()
    # Grab right-axis handle from first panel twin
    h2 = [plt.Line2D([0], [0], color=PALETTE["slate"], lw=1.5, ls=":", marker="o", ms=3.5)]
    l2 = ["Expanding Harrell C-index (right axis)"]
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    _save(fig, fig_dir, "fig2_rolling_prospective_survival")


def make_fig3(res_dir: str, fig_dir: str) -> None:
    """Figure 3: Ridge-Cox Rescue in Low-EPV Regimes, Early Calibration Slope, and ADNI Route A Snapshot Fidelity."""
    df_a = pd.read_csv(os.path.join(res_dir, "trajectory_a_calendar_cutoff.csv"))
    df_f = pd.read_csv(os.path.join(res_dir, "adni_real_snapshot_fidelity.csv"))
    df_a["yr"] = _yr(df_a)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5))

    # Panel A: ADNI_S2 (CN->Imp) Low-EPV Cox std vs Ridge-Cox HR L2 drift & C-index
    ax = axes[0]
    s_std = df_a[(df_a["endpoint_id"] == "ADNI_S2_cn_to_imp") & (df_a["model_type"] == "cox_std")].sort_values("yr")
    s_rid = df_a[(df_a["endpoint_id"] == "ADNI_S2_cn_to_imp") & (df_a["model_type"] == "cox_ridge_cv")].sort_values("yr")
    ax.plot(s_std["yr"], s_std["coef_l2_to_ref"], color=PALETTE["crimson"], lw=2.2, marker="o", ms=4, label="Standard Cox (HR L2 drift)")
    ax.plot(s_rid["yr"], s_rid["coef_l2_to_ref"], color=PALETTE["teal"], lw=2.2, marker="D", ms=4, label="Ridge-Cox CV (HR L2 drift)")
    ax.set_xticks([2008, 2011, 2014, 2017, 2020, 2023, 2026])
    ax.set_xlabel("Calendar Cutoff Year")
    ax.set_ylabel("Standardized log(HR) L2 Drift to 2026")
    ax.set_title("A. Low-EPV Coefficient Rescue (ADNI CN->Imp)")
    ax.legend(frameon=True, loc="upper right", fontsize=7.6)

    # Panel B: Early (2008-2010) mean calibration slope across all 8 endpoints (cox_std vs cox_ridge_cv)
    ax = axes[1]
    early = df_a[df_a["cutoff_label"].isin(["2008", "2009", "2010"])]
    eps = [
        ("ADNI_S1_mci_to_dem", "A-S1"), ("ADNI_S2_cn_to_imp", "A-S2"),
        ("ADNI_S3_cdrsb45", "A-S3"), ("ADNI_S4_faq9", "A-S4"),
        ("NACC_S1_mci_to_dem", "N-S1"), ("NACC_S2_cn_to_imp", "N-S2"),
        ("NACC_S3_cdrsb45", "N-S3"), ("NACC_S4_faq9", "N-S4"),
    ]
    x = np.arange(len(eps))
    w = 0.36
    v_std = [early[(early["endpoint_id"] == e) & (early["model_type"] == "cox_std")]["cal_slope"].mean() for e, _ in eps]
    v_rid = [early[(early["endpoint_id"] == e) & (early["model_type"] == "cox_ridge_cv")]["cal_slope"].mean() for e, _ in eps]
    ax.axhline(1.0, color="black", lw=1.0, ls=":")
    ax.bar(x - w / 2, v_std, width=w, color=PALETTE["coral"], label="Standard Cox")
    ax.bar(x + w / 2, v_rid, width=w, color=PALETTE["teal"], label="Ridge-Cox CV")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in eps], fontsize=8.0)
    ax.set_ylim(-0.05, 1.95)
    ax.set_ylabel("Mean Early (2008-2010) Calibration Slope")
    ax.set_title("B. Early Calibration Slope (All 8 Endpoints)")
    ax.legend(frameon=True, loc="upper right", fontsize=7.6)

    # Panel C: ADNI Route A Real Historical Snapshot Fidelity (36m IPCW Brier)
    ax = axes[2]
    f_std = df_f[df_f["model_type"] == "cox_std"]
    pairs = [
        ("ADNI_S1_mci_to_dem", "ADNIMERGE_2023_09_vs_trunc_2025_07", "real_adnimerge_2023_09", "trunc_adnimerge_2025_07", "S1\n(AM 23v25)"),
        ("ADNI_S3_cdrsb45", "ADNIMERGE_2023_09_vs_trunc_2025_07", "real_adnimerge_2023_09", "trunc_adnimerge_2025_07", "S3\n(AM 23v25)"),
        ("ADNI_S4_faq9", "ADNIMERGE_2023_09_vs_trunc_2025_07", "real_adnimerge_2023_09", "trunc_adnimerge_2025_07", "S4\n(AM 23v25)"),
        ("ADNI_S1_mci_to_dem", "DXSUM_2025_07_vs_trunc_2026_10", "real_dxsum_2025_07", "trunc_dxsum_2026_10_all_phases", "S1\n(DX 25v26)"),
        ("ADNI_S2_cn_to_imp", "DXSUM_2025_07_vs_trunc_2026_10", "real_dxsum_2025_07", "trunc_dxsum_2026_10_all_phases", "S2\n(DX 25v26)"),
    ]
    xf = np.arange(len(pairs))
    r_vals, t_vals = [], []
    for ep, ap, s1, s2, _ in pairs:
        sub = f_std[(f_std["endpoint_id"] == ep) & (f_std["audit_pair"] == ap)]
        r_vals.append(float(sub[sub["source_tag"] == s1]["brier_36m"].iloc[0]))
        t_vals.append(float(sub[sub["source_tag"] == s2]["brier_36m"].iloc[0]))
    b1 = ax.bar(xf - w / 2, r_vals, width=w, color=PALETTE["navy"], label="Real historical release")
    b2 = ax.bar(xf + w / 2, t_vals, width=w, color=PALETTE["gold"], label="Truncated newer release")
    for bi, bars in enumerate((b1, b2)):
        for bar in bars:
            h = bar.get_height()
            x_off = -0.02 if bi == 0 else 0.02
            ax.text(bar.get_x() + bar.get_width() / 2 + x_off, h + 0.004, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=6.6, rotation=90)
    ax.set_xticks(xf)
    ax.set_xticklabels([l for *_, l in pairs], fontsize=7.6)
    ax.set_ylim(0, max(r_vals + t_vals) * 1.34)
    ax.set_ylabel("36-Month IPCW Brier Score")
    ax.set_title("C. ADNI Route A Snapshot Fidelity Audit")
    ax.legend(frameon=True, loc="upper right", fontsize=7.5)

    fig.tight_layout()
    _save(fig, fig_dir, "fig3_ridge_rescue_and_fidelity")


def make_supp_figures(res_dir: str, fig_dir: str) -> None:
    """Supplementary Figures S1 (All 8 Brier A vs C), S2 (Trajectory B doubling), S3 (9-covariate log-HR trajectories)."""
    df_a = pd.read_csv(os.path.join(res_dir, "trajectory_a_calendar_cutoff.csv"))
    df_b = pd.read_csv(os.path.join(res_dir, "trajectory_b_chronological_doubling.csv"))
    df_c = pd.read_csv(os.path.join(res_dir, "trajectory_c_locked_unfreezing.csv"))
    df_a["yr"] = _yr(df_a)
    df_c["yr"] = _yr(df_c)

    all_eps = [
        ("ADNI_S1_mci_to_dem", "ADNI S1: MCI->Dem"),
        ("ADNI_S2_cn_to_imp", "ADNI S2: CN->Imp"),
        ("ADNI_S3_cdrsb45", "ADNI S3: CDR-SB>=4.5"),
        ("ADNI_S4_faq9", "ADNI S4: FAQ>=9"),
        ("NACC_S1_mci_to_dem", "NACC S1: MCI->Dem"),
        ("NACC_S2_cn_to_imp", "NACC S2: CN->Imp"),
        ("NACC_S3_cdrsb45", "NACC S3: CDR-SB>=4.5"),
        ("NACC_S4_faq9", "NACC S4: FAQ>=9"),
    ]

    # Fig S1: All 8 endpoints 36m IPCW Brier Traj A vs Traj C
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.5))
    for ax, (ep, title) in zip(axes.flat, all_eps):
        sa = df_a[(df_a["endpoint_id"] == ep) & (df_a["model_type"] == "cox_std")].sort_values("yr")
        sc = df_c[(df_c["endpoint_id"] == ep) & (df_c["model_type"] == "cox_std")].sort_values("yr")
        sr = df_a[(df_a["endpoint_id"] == ep) & (df_a["model_type"] == "cox_ridge_cv")].sort_values("yr")
        ax.plot(sa["yr"], sa["brier_36m"], color=PALETTE["navy"], lw=2.0, marker="o", ms=3, label="Traj. A (Cox std)")
        ax.plot(sc["yr"], sc["brier_36m"], color=PALETTE["crimson"], lw=1.9, ls="--", marker="s", ms=3, label="Traj. C (Locked <=2008)")
        ax.plot(sr["yr"], sr["brier_36m"], color=PALETTE["teal"], lw=1.6, ls="-.", label="Traj. A (Ridge-Cox CV)")
        ax.set_xticks([2008, 2014, 2020, 2026])
        ax.set_title(title)
        ax.set_xlabel("Cutoff Year")
        ax.set_ylabel("36m IPCW Brier")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, fig_dir, "figS1_all8_brier_abc")

    # Fig S2: Trajectory B chronological doubling across all 8 endpoints
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.5))
    for ax, (ep, title) in zip(axes.flat, all_eps):
        sub = df_b[df_b["endpoint_id"] == ep]
        for mt, clr, lbl in [("cox_std", PALETTE["navy"], "Cox std"), ("cox_ridge_cv", PALETTE["teal"], "Ridge-Cox CV")]:
            s = sub[sub["model_type"] == mt].sort_values("train_n")
            ax.plot(s["train_n"], s["brier_36m"], color=clr, lw=2.0, marker="o", ms=3.5, label=lbl)
            if "brier_36m_p025" in s.columns:
                ax.fill_between(s["train_n"], s["brier_36m_p025"], s["brier_36m_p975"], color=clr, alpha=0.14)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        xmin, xmax = sub["train_n"].min(), sub["train_n"].max()
        ticks = [t for t in [50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000] if xmin * 0.8 <= t <= xmax * 1.2]
        if len(ticks) > 5:
            ticks = ticks[::2]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:,}" if t >= 1000 else str(t) for t in ticks], fontsize=7.8)
        ax.set_title(title)
        ax.set_xlabel("Chronological Train N (log)")
        ax.set_ylabel("36m IPCW Brier")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, fig_dir, "figS2_trajectory_b_doubling_survival")

    # Fig S3: 9-covariate standardized log(HR) trajectories in ADNI_S1 & NACC_S1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8), sharey=True)
    clrs = plt.cm.tab10(np.linspace(0, 0.9, 9))
    for ax, ep, title in [
        (ax1, "ADNI_S1_mci_to_dem", "A. ADNI MCI->Dementia: 9 Standardized log(HR) Trajectories"),
        (ax2, "NACC_S1_mci_to_dem", "B. NACC MCI->Dementia: 9 Standardized log(HR) Trajectories"),
    ]:
        s = df_a[(df_a["endpoint_id"] == ep) & (df_a["model_type"] == "cox_std")].sort_values("yr")
        ax.axhline(0.0, color="black", lw=0.8, ls=":")
        for j, flbl in enumerate(FEAT_LABELS):
            ax.plot(s["yr"], s[f"beta_ref_{j}"], color=clrs[j], lw=1.9, marker="o", ms=3.0, label=flbl)
        ax.set_xticks([2008, 2011, 2014, 2017, 2020, 2023, 2026])
        ax.set_xlabel("Calendar Cutoff Year")
        ax.set_title(title)
    ax1.set_ylabel("Standardized log(HR) (2026 Fixed SD Scale)")
    h, l = ax1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=True, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    _save(fig, fig_dir, "figS3_covariate_log_hr_trajectories")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-dir", default="outputs/survival_full")
    parser.add_argument("--fig-dir", default="outputs/figures")
    args = parser.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    _setup_style()
    make_fig1(args.res_dir, args.fig_dir)
    make_fig2(args.res_dir, args.fig_dir)
    make_fig3(args.res_dir, args.fig_dir)
    make_supp_figures(args.res_dir, args.fig_dir)
    print(f"Generated 6 survival figures in {args.fig_dir}")


if __name__ == "__main__":
    main()
