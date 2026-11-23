"""Programmatically generate zero-overfull booktabs/longtable LaTeX tables from survival pipeline CSVs."""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd


def _fmt(v, dec=3):
    if pd.isna(v) or not np.isfinite(v):
        return "--"
    if v < 0:
        return f"$-${abs(v):.{dec}f}"
    return f"{v:.{dec}f}"


def _fmt_signed(v, dec=1):
    if pd.isna(v) or not np.isfinite(v):
        return "--"
    if v < 0:
        return f"$-${abs(v):.{dec}f}"
    return f"$+${v:.{dec}f}"


def make_tables(res_dir: str, tab_dir: str) -> None:
    os.makedirs(tab_dir, exist_ok=True)
    df_a = pd.read_csv(os.path.join(res_dir, "trajectory_a_calendar_cutoff.csv"))
    df_c = pd.read_csv(os.path.join(res_dir, "trajectory_c_locked_unfreezing.csv"))
    df_r = pd.read_csv(os.path.join(res_dir, "rolling_prospective.csv"))
    df_f = pd.read_csv(os.path.join(res_dir, "adni_real_snapshot_fidelity.csv"))

    ep_order = [
        ("ADNI_S1_mci_to_dem", "ADNI S1: MCI$\\to$Dem"),
        ("ADNI_S2_cn_to_imp", "ADNI S2: CN$\\to$Imp"),
        ("ADNI_S3_cdrsb45", "ADNI S3: CDR-SB$\\ge$4.5"),
        ("ADNI_S4_faq9", "ADNI S4: FAQ$\\ge$9"),
        ("NACC_S1_mci_to_dem", "NACC S1: MCI$\\to$Dem"),
        ("NACC_S2_cn_to_imp", "NACC S2: CN$\\to$Imp"),
        ("NACC_S3_cdrsb45", "NACC S3: CDR-SB$\\ge$4.5"),
        ("NACC_S4_faq9", "NACC S4: FAQ$\\ge$9"),
    ]

    # Table 1: Three-Trajectory Causal Decomposition (A/C(2008) -> C(2026) -> A(2026), cox_std)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Three-trajectory causal decomposition across eight dual-cohort Alzheimer's disease survival endpoints (2008--2026, exact Efron standard Cox): separating pure administrative censoring maturation ($C_{2008} \to C_{2026}$) from subsequent protocol accrual and case-mix shift ($C_{2026} \to A_{2026}$).}",
        r"\label{tab:lc01_decomposition}",
        r"\ManuscriptTableSize{7.8}{9.6}",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{@{}lrrrrrrrrr@{}}",
        r"\toprule",
        r"Endpoint & Eval $N$ (Ev.) & $A_{08}$ E/O & $C_{26}$ E/O & $A_{26}$ E/O & $A_{08}$ BS36 & $C_{26}$ BS36 & $A_{26}$ BS36 & $A_{26}$ $C_{\text{H}}$ & $C_{26}$ $L_2$ \\",
        r"\midrule",
    ]
    for ei, (ep, lbl) in enumerate(ep_order):
        if ei == 4:
            lines.append(r"\midrule")
        sa = df_a[(df_a["endpoint_id"] == ep) & (df_a["model_type"] == "cox_std")]
        sc = df_c[(df_c["endpoint_id"] == ep) & (df_c["model_type"] == "cox_std")]
        a08 = sa[sa["cutoff_label"] == "2008"].iloc[0]
        c26 = sc[sc["cutoff_label"] == "2026_final"].iloc[0]
        a26 = sa[sa["cutoff_label"] == "2026_final"].iloc[0]
        lines.append(
            f"{lbl} & {int(a26['eval_n']):,} ({int(a26['eval_events']):,}) & "
            f"{_fmt(a08['oe_ratio_36m'], 3)} & {_fmt(c26['oe_ratio_36m'], 3)} & {_fmt(a26['oe_ratio_36m'], 3)} & "
            f"{_fmt(a08['brier_36m'], 4)} & {_fmt(c26['brier_36m'], 4)} & {_fmt(a26['brier_36m'], 4)} & "
            f"{_fmt(a26['c_harrell'], 3)} & {_fmt(c26['coef_l2_to_ref'], 3)} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"\textit{Notes.} $A_{08} = C_{08}$ denotes the 2008 calendar cutoff; $C_{26}$ freezes the $\le$2008 cohort (zero new recruits) and unfreezes follow-up to 2026; $A_{26}$ is the full 2026 cohort. BS36, 36-month IPCW Brier score; $C_{\text{H}}$, Harrell's C-index; HR L2, Euclidean distance of fixed-SD standardized $\log(\text{HR})$ vector from $A_{26}$.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    with open(os.path.join(tab_dir, "table1_decomposition.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # Table 2: Early Cox_std vs Ridge-Cox CV + Rolling Prospective Expanding vs Sliding summary
    lines2 = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Early-cutoff (2008--2010) regularization rescue (standard Cox vs.\ 5-fold CV Ridge-Cox) and rolling prospective deployment summary (2009--2022 expanding vs.\ 5-year sliding window).}",
        r"\label{tab:lc01_ridge_rolling}",
        r"\ManuscriptTableSize{7.8}{9.6}",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}",
        r"\toprule",
        r"Endpoint & Std Sl.$^{\text{a}}$ & Rid.\ Sl.$^{\text{a}}$ & Std $L_2^{\text{a}}$ & Rid.\ $L_2^{\text{a}}$ & Exp.\ E/O$^{\text{b}}$ & Sli.\ E/O$^{\text{b}}$ & Exp.\ $C_{\text{H}}^{\text{b}}$ & Worst Bias$^{\text{c}}$ \\",
        r"\midrule",
    ]
    early = df_a[df_a["cutoff_label"].isin(["2008", "2009", "2010"])]
    for ei, (ep, lbl) in enumerate(ep_order):
        if ei == 4:
            lines2.append(r"\midrule")
        es = early[(early["endpoint_id"] == ep) & (early["model_type"] == "cox_std")]
        er = early[(early["endpoint_id"] == ep) & (early["model_type"] == "cox_ridge_cv")]
        re = df_r[(df_r["endpoint_id"] == ep) & (df_r["model_type"] == "cox_std") & (df_r["window_strategy"] == "expanding")]
        rs = df_r[(df_r["endpoint_id"] == ep) & (df_r["model_type"] == "cox_std") & (df_r["window_strategy"] == "sliding_5y")]
        worst_b = re.loc[re["risk_bias_36m_pct"].abs().idxmax(), "risk_bias_36m_pct"] if len(re) > 0 else np.nan
        lines2.append(
            f"{lbl} & {_fmt(es['cal_slope'].mean(), 3)} & {_fmt(er['cal_slope'].mean(), 3)} & "
            f"{_fmt(es['coef_l2_to_ref'].mean(), 3)} & {_fmt(er['coef_l2_to_ref'].mean(), 3)} & "
            f"{_fmt(re['oe_ratio_36m'].mean(), 3)} & {_fmt(rs['oe_ratio_36m'].mean() if len(rs) > 0 else np.nan, 3)} & "
            f"{_fmt(re['c_harrell'].mean(), 3)} & {_fmt_signed(worst_b, 1)} \\\\"
        )
    lines2 += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\begin{minipage}{\linewidth}\footnotesize",
        r"\textit{Notes.} $^{\text{a}}$Averaged across early annual cutoffs 2008--2010. $^{\text{b}}$Averaged across prospective deployment years 2009--2022 (future enrollees $\text{bl\_dt} > T_k$). $^{\text{c}}$Largest signed 36-month risk bias (percentage points) under expanding-window prospective deployment.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    with open(os.path.join(tab_dir, "table2_ridge_rolling.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines2) + "\n")

    # Table S1: ADNI Route A Real Historical Snapshot Fidelity Audit
    fs = df_f[df_f["model_type"] == "cox_std"]
    lines_s1 = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{ADNI Route A historical release fidelity audit: survival models fitted on real archived releases versus calendar-truncated newer releases (exact Efron standard Cox).}",
        r"\label{tab:lc01_s1_fidelity}",
        r"\ManuscriptTableSize{7.8}{9.6}",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{@{}lllrrrrrr@{}}",
        r"\toprule",
        r"Audit Pair & Endpoint & Source & Train $N$ & Ev. & $C_{\text{H}}$ & $C_{\text{Uno}}$ & BS36 & E/O 36m \\",
        r"\midrule",
    ]
    amap = {
        "ADNIMERGE_2023_09_vs_trunc_2025_07": "AM 23-09 vs 25-07",
        "DXSUM_2025_07_vs_trunc_2026_10": "DX 25-07 vs 26-10",
    }
    smap = {
        "real_adnimerge_2023_09": "Real 2023-09",
        "trunc_adnimerge_2025_07": "Trunc.\\ 2025-07",
        "real_dxsum_2025_07": "Real 2025-07",
        "trunc_dxsum_2026_10_all_phases": "Trunc.\\ 26-10 (All)",
        "trunc_dxsum_2026_10_adni1_3_only": "Trunc.\\ 26-10 (ADNI1--3)",
    }
    for _, r in fs.iterrows():
        ep_s = r["endpoint_id"].replace("ADNI_", "").replace("_", r"\_")
        lines_s1.append(
            f"{amap[r['audit_pair']]} & {ep_s} & {smap[r['source_tag']]} & "
            f"{int(r['train_n']):,} & {int(r['train_events']):,} & "
            f"{_fmt(r['c_harrell'], 4)} & {_fmt(r['c_uno_48m'], 4)} & "
            f"{_fmt(r['brier_36m'], 4)} & {_fmt(r['oe_ratio_36m'], 4)} \\\\"
        )
    lines_s1 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(tab_dir, "tableS1_fidelity.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines_s1) + "\n")

    # Table S2: Complete 2026 Final Reference Performance across all 8 endpoints x 2 models
    fin = df_a[df_a["cutoff_label"] == "2026_final"]
    lines_s2 = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Final 2026 full-cohort reference survival performance across all eight endpoints and both Cox estimators on the independent 30\% holdout pool.}",
        r"\label{tab:lc01_s2_final}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\footnotesize",
        r"\begin{tabular}{@{}llrrrrrrr@{}}",
        r"\toprule",
        r"Endpoint & Estimator & Dev $N$ (Ev.) & $C_{\text{H}}$ & $C_{\text{Uno}}$ & BS36 & IBS$_{12\text{--}60}$ & E/O 36m & Slope \\",
        r"\midrule",
    ]
    for ei, (ep, lbl) in enumerate(ep_order):
        if ei == 4:
            lines_s2.append(r"\midrule")
        for mi, (mt, mlbl) in enumerate([("cox_std", "Std Cox"), ("cox_ridge_cv", "Ridge-Cox")]):
            r = fin[(fin["endpoint_id"] == ep) & (fin["model_type"] == mt)].iloc[0]
            e_str = lbl if mi == 0 else ""
            lines_s2.append(
                f"{e_str} & {mlbl} & {int(r['train_n']):,} ({int(r['train_events']):,}) & "
                f"{_fmt(r['c_harrell'], 4)} & {_fmt(r['c_uno_48m'], 4)} & "
                f"{_fmt(r['brier_36m'], 4)} & {_fmt(r['ibs_12_60m'], 4)} & "
                f"{_fmt(r['oe_ratio_36m'], 3)} & {_fmt(r['cal_slope'], 3)} \\\\"
            )
    lines_s2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(tab_dir, "tableS2_final_reference.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines_s2) + "\n")

    print(f"Generated 4 survival LaTeX tables in {tab_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res-dir", default="outputs/survival_full")
    parser.add_argument("--tab-dir", default="outputs/tables")
    args = parser.parse_args()
    make_tables(args.res_dir, args.tab_dir)


if __name__ == "__main__":
    main()
