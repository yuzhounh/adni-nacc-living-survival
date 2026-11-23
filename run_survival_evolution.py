"""End-to-end dual-cohort survival evolution pipeline (Trajectories A/B/C + Rolling Prospective + Route A Fidelity Audit)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from build_cohorts import (
    EndpointBundle,
    FEATURES_9,
    build_adni_endpoints,
    build_nacc_endpoints,
    compute_survival_at_cutoff,
    load_adnimerge_snapshot_df,
)
from survival_engine import (
    compute_eval_censoring_km,
    eval_step_surv,
    evaluate_survival_model,
    fit_cox_model,
    km_survival_curve,
)


ANNUAL_CUTOFF_YEARS = list(range(2008, 2026))
ROLLING_DEPLOY_YEARS = list(range(2009, 2023))


def make_doubling_grid(n_full: int, min_target: int) -> List[int]:
    """Generate halving-from-full size ladder down to nearest power-of-2 target around min_target."""
    sizes = [n_full]
    k = 1
    while True:
        val = int(np.floor(n_full / (2 ** k) + 0.5))
        if val < max(30, int(min_target * 0.7)):
            break
        if val not in sizes:
            sizes.append(val)
        k += 1
    return sorted(sizes)


def run_pilot_audits(bundles: List[EndpointBundle], model_types: Tuple[str, ...]) -> Dict:
    """Strict pilot audit: 0 identity leakage, strict temporal cutoff, lifelines < 1e-5 check, anchor zero-drift."""
    report = {"checks": [], "all_passed": True}
    for b in bundles:
        full_df = compute_survival_at_cutoff(b.baseline_df, b.long_df, b.final_cutoff)
        dev_full = full_df[~full_df["is_eval"]].copy()
        ev_full = full_df[full_df["is_eval"]].copy()

        # 1. Zero identity overlap
        overlap = len(set(dev_full["subj_id"]).intersection(set(ev_full["subj_id"])))
        assert overlap == 0, f"{b.endpoint_id}: dev/eval overlap = {overlap}"

        # 2. Strict temporal cutoff check at 2012-12-31
        cut_test = pd.Timestamp("2012-12-31")
        df_2012 = compute_survival_at_cutoff(b.baseline_df[~b.baseline_df["is_eval"]], b.long_df, cut_test)
        assert (df_2012["bl_dt"] <= cut_test).all()
        max_possible_m = ((cut_test - df_2012["bl_dt"]).dt.days / 30.4375) + 1e-6
        assert (df_2012["time_m"] <= max_possible_m).all(), f"{b.endpoint_id}: future follow-up leaked!"

        # 3. Cross-verify cox_std vs lifelines.CoxPHFitter (< 1e-5); for huge NACC sets (>3000),
        # verify exact numerical agreement on deterministic first 3,000 subjects (full set still fitted & checked by ours)
        X_dev = dev_full[FEATURES_9].to_numpy(dtype=np.float64)
        t_dev = dev_full["time_m"].to_numpy(dtype=np.float64)
        e_dev = dev_full["event"].to_numpy(dtype=np.int64)
        m_full = fit_cox_model(X_dev, t_dev, e_dev, model_type="cox_std", lambda_std=0.01)
        assert m_full.converged, f"{b.endpoint_id}: cox_std did not converge on full dev"

        n_chk = min(len(dev_full), 3000)
        X_chk, t_chk, e_chk = X_dev[:n_chk], t_dev[:n_chk], e_dev[:n_chk]
        m_ours = fit_cox_model(X_chk, t_chk, e_chk, model_type="cox_std", lambda_std=0.01)
        assert m_ours.converged, f"{b.endpoint_id}: cox_std did not converge on audit slice"

        X_imp = X_chk.copy()
        for j in range(X_imp.shape[1]):
            bad = ~np.isfinite(X_imp[:, j])
            X_imp[bad, j] = m_ours.medians[j]
        X_std = (X_imp - m_ours.mean_x) / m_ours.sd_x
        df_ll = pd.DataFrame(X_std, columns=FEATURES_9)
        df_ll["T"] = t_chk
        df_ll["E"] = e_chk
        cph = CoxPHFitter(penalizer=0.01 / len(df_ll), l1_ratio=0.0)
        cph.fit(df_ll, duration_col="T", event_col="E", fit_options={"step_size": 0.5, "precision": 1e-9})
        max_beta_diff = float(np.max(np.abs(m_ours.beta - cph.params_.values)))
        assert max_beta_diff < 1e-5, f"{b.endpoint_id}: lifelines beta diff {max_beta_diff}"

        # 4. Censoring KM support check G_eval(48m)
        g_ut, g_s = compute_eval_censoring_km(
            ev_full["time_m"].to_numpy(dtype=np.float64), ev_full["event"].to_numpy(dtype=np.int64)
        )
        g_48 = float(eval_step_surv(g_ut, g_s, np.array([48.0]))[0])
        g_60 = float(eval_step_surv(g_ut, g_s, np.array([60.0]))[0])

        report["checks"].append({
            "endpoint_id": b.endpoint_id,
            "dev_n": int(len(dev_full)),
            "dev_events": int(e_dev.sum()),
            "eval_n": int(len(ev_full)),
            "eval_events": int(ev_full["event"].sum()),
            "lifelines_max_beta_diff": max_beta_diff,
            "eval_G_48m": round(g_48, 4),
            "eval_G_60m": round(g_60, 4),
            "overlap": overlap,
        })
        print(
            f"  [AUDIT PASS] {b.endpoint_id:20s}: dev={len(dev_full):5d}({int(e_dev.sum()):4d} ev), "
            f"eval={len(ev_full):5d}({int(ev_full['event'].sum()):4d} ev), "
            f"lifelines_diff={max_beta_diff:.2e}, G(48m)={g_48:.3f}, G(60m)={g_60:.3f}"
        )
    return report


def run_three_trajectories_for_endpoint(
    b: EndpointBundle, n_reps_b: int, rng_seed: int
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """Execute Trajectory A (calendar cutoffs), B (chronological doubling), C (locked <=2008 unfreezing)."""
    full_df = compute_survival_at_cutoff(b.baseline_df, b.long_df, b.final_cutoff)
    dev_full = full_df[~full_df["is_eval"]].copy().reset_index(drop=True)
    ev_full = full_df[full_df["is_eval"]].copy().reset_index(drop=True)

    X_eval = ev_full[FEATURES_9].to_numpy(dtype=np.float64)
    t_eval = ev_full["time_m"].to_numpy(dtype=np.float64)
    e_eval = ev_full["event"].to_numpy(dtype=np.int64)
    g_ut, g_surv = compute_eval_censoring_km(t_eval, e_eval)
    km_ut, km_surv = km_survival_curve(t_eval, e_eval)

    # Fit final reference models to obtain fixed ref_sd and ref_beta per model_type
    X_dev_full = dev_full[FEATURES_9].to_numpy(dtype=np.float64)
    t_dev_full = dev_full["time_m"].to_numpy(dtype=np.float64)
    e_dev_full = dev_full["event"].to_numpy(dtype=np.int64)

    ref_models = {}
    ref_sd = None
    for mtype in ("cox_std", "cox_ridge_cv"):
        m_ref = fit_cox_model(X_dev_full, t_dev_full, e_dev_full, model_type=mtype, cv_seed=rng_seed)
        ref_models[mtype] = m_ref
        if ref_sd is None:
            ref_sd = m_ref.sd_x.copy()

    # Build cutoffs list: 2008-12-31 .. 2025-12-31 + final_cutoff
    cutoffs = [pd.Timestamp(f"{yr}-12-31") for yr in ANNUAL_CUTOFF_YEARS] + [b.final_cutoff]
    cutoff_labels = [str(yr) for yr in ANNUAL_CUTOFF_YEARS] + ["2026_final"]

    # Locked early cohort IDs: dev subjects with bl_dt <= 2008-12-31
    dev_bl = b.baseline_df[~b.baseline_df["is_eval"]].copy()
    locked_2008_ids = set(dev_bl.loc[dev_bl["bl_dt"] <= pd.Timestamp("2008-12-31"), "subj_id"])

    rows_a: List[Dict] = []
    rows_c: List[Dict] = []

    for cut_dt, cut_lbl in zip(cutoffs, cutoff_labels):
        # Trajectory A: calendar cutoff on all dev subjects
        tr_a = compute_survival_at_cutoff(dev_bl, b.long_df, cut_dt)
        # Trajectory C: locked <= 2008 dev cohort unfrozen to cut_dt
        tr_c = compute_survival_at_cutoff(dev_bl, b.long_df, cut_dt, allowed_ids=locked_2008_ids)

        for traj_name, tr_df, bucket in [("A_calendar_cutoff", tr_a, rows_a), ("C_locked_unfreezing", tr_c, rows_c)]:
            n_tr = len(tr_df)
            ev_tr = int(tr_df["event"].sum()) if n_tr > 0 else 0
            med_fu = float(tr_df["time_m"].median()) if n_tr > 0 else float("nan")
            mean_cdrsb = float(tr_df["cdrsb"].mean()) if n_tr > 0 else float("nan")
            X_tr = tr_df[FEATURES_9].to_numpy(dtype=np.float64) if n_tr > 0 else np.empty((0, len(FEATURES_9)))
            t_tr = tr_df["time_m"].to_numpy(dtype=np.float64) if n_tr > 0 else np.empty(0)
            e_tr = tr_df["event"].to_numpy(dtype=np.int64) if n_tr > 0 else np.empty(0, dtype=np.int64)

            for mtype in ("cox_std", "cox_ridge_cv"):
                m = fit_cox_model(X_tr, t_tr, e_tr, model_type=mtype, cv_seed=rng_seed)
                metrics = evaluate_survival_model(
                    m, X_eval, t_eval, e_eval, g_ut, g_surv, km_ut, km_surv,
                    ref_sd=ref_sd, ref_beta_std=ref_models[mtype].beta * (ref_sd / ref_models[mtype].sd_x),
                )
                row = {
                    "cohort": b.cohort,
                    "endpoint_id": b.endpoint_id,
                    "trajectory": traj_name,
                    "model_type": mtype,
                    "cutoff_label": cut_lbl,
                    "cutoff_date": str(cut_dt.date()),
                    "train_n": int(n_tr),
                    "train_events": int(ev_tr),
                    "train_event_rate_pct": round(float(ev_tr / n_tr * 100.0), 2) if n_tr > 0 else float("nan"),
                    "train_median_fu_m": round(med_fu, 2) if np.isfinite(med_fu) else float("nan"),
                    "train_mean_cdrsb": round(mean_cdrsb, 3) if np.isfinite(mean_cdrsb) else float("nan"),
                    "eval_n": int(len(ev_full)),
                    "eval_events": int(e_eval.sum()),
                }
                row.update(metrics)
                bucket.append(row)

    # Trajectory B: chronological doubling on mature final dev_full sorted by exact bl_dt
    min_target = 50 if b.cohort == "ADNI" else 100
    sizes = make_doubling_grid(len(dev_full), min_target=min_target)
    # Group indices by exact bl_dt to permute only tied dates
    dev_sorted = dev_full.sort_values(["bl_dt", "subj_id"]).reset_index(drop=True)
    date_vals = dev_sorted["bl_dt"].to_numpy()
    X_bs = dev_sorted[FEATURES_9].to_numpy(dtype=np.float64)
    t_bs = dev_sorted["time_m"].to_numpy(dtype=np.float64)
    e_bs = dev_sorted["event"].to_numpy(dtype=np.int64)

    # Precompute date tie groups
    uniq_dates, first_idx, counts = np.unique(date_vals, return_index=True, return_counts=True)
    tie_groups = [(int(s), int(s + c)) for s, c in zip(first_idx, counts) if c > 1]

    rng = np.random.default_rng(rng_seed + 101)
    rows_b: List[Dict] = []

    for sz in sizes:
        is_full = (sz == len(dev_sorted))
        boundary_in_tie = False if is_full else (date_vals[sz - 1] == date_vals[sz])
        reps_here = 1 if (is_full or not boundary_in_tie) else (min(n_reps_b, 20) if sz >= 3000 else n_reps_b)

        for mtype in ("cox_std", "cox_ridge_cv"):
            rep_metrics = []
            rep_evs = []
            for r in range(reps_here):
                if is_full:
                    idx_sub = np.arange(len(dev_sorted))
                elif not boundary_in_tie:
                    idx_sub = np.arange(sz)
                else:
                    perm = np.arange(len(dev_sorted))
                    for s_g, e_g in tie_groups:
                        if s_g < sz < e_g:
                            perm[s_g:e_g] = rng.permutation(perm[s_g:e_g])
                    idx_sub = np.sort(perm[:sz])

                # Canonical subject-ID order (dev_sorted is ordered by bl_dt, subj_id; sort by subj_id for exact match at N_full)
                if is_full:
                    X_sub, t_sub, e_sub = X_dev_full, t_dev_full, e_dev_full
                else:
                    sub_ids = dev_sorted["subj_id"].iloc[idx_sub].to_numpy()
                    ord_id = np.argsort(sub_ids)
                    idx_ord = idx_sub[ord_id]
                    X_sub, t_sub, e_sub = X_bs[idx_ord], t_bs[idx_ord], e_bs[idx_ord]

                m = fit_cox_model(X_sub, t_sub, e_sub, model_type=mtype, cv_seed=rng_seed)
                met = evaluate_survival_model(
                    m, X_eval, t_eval, e_eval, g_ut, g_surv, km_ut, km_surv,
                    ref_sd=ref_sd, ref_beta_std=ref_models[mtype].beta * (ref_sd / ref_models[mtype].sd_x),
                )
                rep_metrics.append(met)
                rep_evs.append(int(np.sum(e_sub)))

            df_r = pd.DataFrame(rep_metrics)
            row = {
                "cohort": b.cohort,
                "endpoint_id": b.endpoint_id,
                "trajectory": "B_chronological_doubling",
                "model_type": mtype,
                "train_n": int(sz),
                "n_reps_run": int(reps_here),
                "boundary_tied": int(boundary_in_tie),
                "max_bl_date": str(pd.Timestamp(date_vals[sz - 1]).date()),
                "train_events_mean": round(float(np.mean(rep_evs)), 2),
                "eval_n": int(len(ev_full)),
                "eval_events": int(e_eval.sum()),
            }
            for col in [
                "c_harrell", "c_uno_48m", "brier_36m", "brier_60m", "ibs_12_60m",
                "oe_ratio_36m", "risk_bias_36m_pct", "oe_ratio_60m", "risk_bias_60m_pct",
                "cal_slope", "coef_l2_to_ref", "coef_cos_to_ref", "lambda_pen",
            ]:
                vals = df_r[col].to_numpy(dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                row[col] = float(np.mean(vals)) if len(vals) > 0 else float("nan")
                row[f"{col}_p025"] = float(np.percentile(vals, 2.5)) if len(vals) > 0 else float("nan")
                row[f"{col}_p975"] = float(np.percentile(vals, 97.5)) if len(vals) > 0 else float("nan")
            rows_b.append(row)

    # Verify zero-drift anchors: B(N_full) == A(2026_final) and C(2008) == A(2008)
    anchor_check = {}
    for mtype in ("cox_std", "cox_ridge_cv"):
        a_2008 = [r for r in rows_a if r["model_type"] == mtype and r["cutoff_label"] == "2008"][0]
        c_2008 = [r for r in rows_c if r["model_type"] == mtype and r["cutoff_label"] == "2008"][0]
        a_fin = [r for r in rows_a if r["model_type"] == mtype and r["cutoff_label"] == "2026_final"][0]
        b_fin = [r for r in rows_b if r["model_type"] == mtype and r["train_n"] == len(dev_full)][0]
        d_ac = abs(a_2008["c_harrell"] - c_2008["c_harrell"]) + abs(a_2008["brier_36m"] - c_2008["brier_36m"])
        d_ab = abs(a_fin["c_harrell"] - b_fin["c_harrell"]) + abs(a_fin["brier_36m"] - b_fin["brier_36m"])
        assert d_ac < 1e-10, f"{b.endpoint_id} {mtype}: C(2008) != A(2008) drift={d_ac}"
        assert d_ab < 1e-10, f"{b.endpoint_id} {mtype}: B(N_full) != A(2026_final) drift={d_ab}"
        anchor_check[mtype] = {"C2008_vs_A2008": d_ac, "Bfull_vs_Afinal": d_ab}

    return rows_a, rows_b, rows_c, anchor_check


def run_rolling_prospective_for_endpoint(b: EndpointBundle, rng_seed: int) -> List[Dict]:
    """Rolling prospective deployment: train on <= T_k (expanding & 5y sliding), evaluate on ALL future enrollees bl_dt > T_k."""
    full_all = compute_survival_at_cutoff(b.baseline_df, b.long_df, b.final_cutoff)
    rows: List[Dict] = []

    for yr in ROLLING_DEPLOY_YEARS:
        cut_dt = pd.Timestamp(f"{yr}-12-31")
        # Future enrollees (strictly bl_dt > cut_dt) evaluated on final mature follow-up
        fut_df = full_all[full_all["bl_dt"] > cut_dt].copy().reset_index(drop=True)
        if len(fut_df) < 50 or int(fut_df["event"].sum()) < 15:
            continue

        # Training pool strictly reconstructed at cut_dt (expanding window: all bl_dt <= cut_dt)
        tr_exp = compute_survival_at_cutoff(b.baseline_df, b.long_df, cut_dt)
        assert (tr_exp["bl_dt"] <= cut_dt).all()
        assert (fut_df["bl_dt"] > cut_dt).all()
        assert len(set(tr_exp["subj_id"]).intersection(set(fut_df["subj_id"]))) == 0

        # Recent 5-year sliding window: cut_dt - 5y < bl_dt <= cut_dt
        slide_start = pd.Timestamp(f"{yr - 5}-12-31")
        tr_slide = tr_exp[tr_exp["bl_dt"] > slide_start].copy().reset_index(drop=True)

        X_fut = fut_df[FEATURES_9].to_numpy(dtype=np.float64)
        t_fut = fut_df["time_m"].to_numpy(dtype=np.float64)
        e_fut = fut_df["event"].to_numpy(dtype=np.int64)
        g_ut, g_surv = compute_eval_censoring_km(t_fut, e_fut)
        km_ut, km_surv = km_survival_curve(t_fut, e_fut)

        # Reference SD from full_all for standardized beta comparability
        ref_sd = np.nanstd(full_all[FEATURES_9].to_numpy(dtype=np.float64), axis=0)
        ref_sd = np.where(ref_sd > 1e-12, ref_sd, 1.0)

        for win_name, tr_df in [("expanding", tr_exp), ("sliding_5y", tr_slide)]:
            if len(tr_df) < 40 or int(tr_df["event"].sum()) < 15:
                continue
            X_tr = tr_df[FEATURES_9].to_numpy(dtype=np.float64)
            t_tr = tr_df["time_m"].to_numpy(dtype=np.float64)
            e_tr = tr_df["event"].to_numpy(dtype=np.int64)

            for mtype in ("cox_std", "cox_ridge_cv"):
                m = fit_cox_model(X_tr, t_tr, e_tr, model_type=mtype, cv_seed=rng_seed)
                met = evaluate_survival_model(
                    m, X_fut, t_fut, e_fut, g_ut, g_surv, km_ut, km_surv, ref_sd=ref_sd, ref_beta_std=None
                )
                row = {
                    "cohort": b.cohort,
                    "endpoint_id": b.endpoint_id,
                    "deploy_year": int(yr),
                    "cutoff_date": str(cut_dt.date()),
                    "window_strategy": win_name,
                    "model_type": mtype,
                    "train_n": int(len(tr_df)),
                    "train_events": int(e_tr.sum()),
                    "train_event_rate_pct": round(float(e_tr.mean() * 100.0), 2),
                    "train_median_fu_m": round(float(np.median(t_tr)), 2),
                    "future_n": int(len(fut_df)),
                    "future_events": int(e_fut.sum()),
                    "future_event_rate_pct": round(float(e_fut.mean() * 100.0), 2),
                }
                row.update(met)
                rows.append(row)
    return rows


def run_adni_real_snapshot_fidelity_audit(
    adni_raw_dir: str, adni_extract_dir: str, adni_bundles: List[EndpointBundle]
) -> List[Dict]:
    """Compare survival models trained on REAL historical ADNI snapshots vs calendar-truncated newer releases."""
    am23_candidates = sorted(glob.glob(os.path.join(adni_raw_dir, "ADNIMERGE*.csv")))
    if not am23_candidates:
        raise FileNotFoundError(f"No ADNIMERGE*.csv found under {adni_raw_dir}")
    am23_path = am23_candidates[-1]
    am25_path = os.path.join(adni_extract_dir, "v2025_07", "csv", "adnimerge.csv")
    dx25_path = os.path.join(adni_extract_dir, "v2025_07", "csv", "dxsum.csv")
    dx26_path = os.path.join(adni_extract_dir, "v2026_10", "csv", "DXSUM.csv")

    b_map = {b.endpoint_id: b for b in adni_bundles}
    rows: List[Dict] = []

    # Audit Pair 1: ADNIMERGE 2023-09 (real) vs ADNIMERGE 2025-07 truncated <= 2023-08-31
    am23 = load_adnimerge_snapshot_df(am23_path)
    am25 = load_adnimerge_snapshot_df(am25_path)
    cut1 = pd.Timestamp("2023-08-31")

    for ep_id, ev_col, cond_fn in [
        ("ADNI_S1_mci_to_dem", "dx_num", lambda s: s == 3),
        ("ADNI_S3_cdrsb45", "cdrsb", lambda s: s >= 4.5),
        ("ADNI_S4_faq9", "faq", lambda s: s >= 9.0),
    ]:
        b = b_map[ep_id]
        full_eval = compute_survival_at_cutoff(b.baseline_df[b.baseline_df["is_eval"]], b.long_df, b.final_cutoff)
        X_eval = full_eval[FEATURES_9].to_numpy(dtype=np.float64)
        t_eval = full_eval["time_m"].to_numpy(dtype=np.float64)
        e_eval = full_eval["event"].to_numpy(dtype=np.int64)
        g_ut, g_surv = compute_eval_censoring_km(t_eval, e_eval)
        km_ut, km_surv = km_survival_curve(t_eval, e_eval)
        dev_bl = b.baseline_df[~b.baseline_df["is_eval"]]
        ref_sd = np.nanstd(dev_bl[FEATURES_9].to_numpy(dtype=np.float64), axis=0)

        for tag, snap_df in [("real_adnimerge_2023_09", am23), ("trunc_adnimerge_2025_07", am25)]:
            lg = snap_df.loc[snap_df[ev_col].notna(), ["subj_id", "visit_dt"]].assign(
                is_event=cond_fn(snap_df.loc[snap_df[ev_col].notna(), ev_col])
            )
            tr = compute_survival_at_cutoff(dev_bl, lg, cut1)
            X_tr = tr[FEATURES_9].to_numpy(dtype=np.float64)
            t_tr = tr["time_m"].to_numpy(dtype=np.float64)
            e_tr = tr["event"].to_numpy(dtype=np.int64)
            for mtype in ("cox_std", "cox_ridge_cv"):
                m = fit_cox_model(X_tr, t_tr, e_tr, model_type=mtype, cv_seed=20261123)
                met = evaluate_survival_model(m, X_eval, t_eval, e_eval, g_ut, g_surv, km_ut, km_surv, ref_sd=ref_sd)
                row = {
                    "audit_pair": "ADNIMERGE_2023_09_vs_trunc_2025_07",
                    "cutoff_date": "2023-08-31",
                    "endpoint_id": ep_id,
                    "source_tag": tag,
                    "model_type": mtype,
                    "train_n": int(len(tr)),
                    "train_events": int(e_tr.sum()),
                }
                row.update(met)
                rows.append(row)

    # Audit Pair 2: DXSUM 2025-07 (real, ADNI1-3 only) vs DXSUM 2026-10 truncated <= 2025-06-30 (all vs ADNI1-3)
    def _norm_dxsum(path: str) -> pd.DataFrame:
        d = pd.read_csv(path, low_memory=False)
        d["RID"] = pd.to_numeric(d["RID"], errors="coerce")
        d["visit_dt"] = pd.to_datetime(d["EXAMDATE"], errors="coerce")
        s = d["DIAGNOSIS"].astype(str).str.strip()
        mp = {"1": 1, "1.0": 1, "CN": 1, "2": 2, "2.0": 2, "MCI": 2, "3": 3, "3.0": 3, "Dementia": 3}
        d["dx_num"] = s.map(mp)
        d = d.dropna(subset=["RID", "visit_dt", "dx_num"]).copy()
        d["subj_id"] = d["RID"].astype(int).map(lambda r: f"ADNI_{r:05d}")
        return d

    dx25 = _norm_dxsum(dx25_path)
    dx26 = _norm_dxsum(dx26_path)
    cut2 = pd.Timestamp("2025-06-30")
    adni13_phases = {"ADNI1", "ADNIGO", "ADNI2", "ADNI3"}

    for ep_id, cond_fn in [
        ("ADNI_S1_mci_to_dem", lambda s: s == 3),
        ("ADNI_S2_cn_to_imp", lambda s: s.isin([2, 3])),
    ]:
        b = b_map[ep_id]
        full_eval = compute_survival_at_cutoff(b.baseline_df[b.baseline_df["is_eval"]], b.long_df, b.final_cutoff)
        X_eval = full_eval[FEATURES_9].to_numpy(dtype=np.float64)
        t_eval = full_eval["time_m"].to_numpy(dtype=np.float64)
        e_eval = full_eval["event"].to_numpy(dtype=np.int64)
        g_ut, g_surv = compute_eval_censoring_km(t_eval, e_eval)
        km_ut, km_surv = km_survival_curve(t_eval, e_eval)
        dev_bl = b.baseline_df[~b.baseline_df["is_eval"]]
        dev_bl_13 = dev_bl[dev_bl["phase_ver"].isin(adni13_phases)]
        ref_sd = np.nanstd(dev_bl[FEATURES_9].to_numpy(dtype=np.float64), axis=0)

        for tag, snap_df, bl_use in [
            ("real_dxsum_2025_07", dx25, dev_bl),
            ("trunc_dxsum_2026_10_all_phases", dx26, dev_bl),
            ("trunc_dxsum_2026_10_adni1_3_only", dx26, dev_bl_13),
        ]:
            lg = snap_df[["subj_id", "visit_dt"]].assign(is_event=cond_fn(snap_df["dx_num"]))
            tr = compute_survival_at_cutoff(bl_use, lg, cut2)
            X_tr = tr[FEATURES_9].to_numpy(dtype=np.float64)
            t_tr = tr["time_m"].to_numpy(dtype=np.float64)
            e_tr = tr["event"].to_numpy(dtype=np.int64)
            for mtype in ("cox_std", "cox_ridge_cv"):
                m = fit_cox_model(X_tr, t_tr, e_tr, model_type=mtype, cv_seed=20261123)
                met = evaluate_survival_model(m, X_eval, t_eval, e_eval, g_ut, g_surv, km_ut, km_surv, ref_sd=ref_sd)
                row = {
                    "audit_pair": "DXSUM_2025_07_vs_trunc_2026_10",
                    "cutoff_date": "2025-06-30",
                    "endpoint_id": ep_id,
                    "source_tag": tag,
                    "model_type": mtype,
                    "train_n": int(len(tr)),
                    "train_events": int(e_tr.sum()),
                }
                row.update(met)
                rows.append(row)

    return rows


def _worker_endpoint(payload):
    idx, b, n_reps = payload
    ts = time.time()
    ra, rb, rc, anch = run_three_trajectories_for_endpoint(b, n_reps_b=n_reps, rng_seed=20261123 + idx * 17)
    rr = run_rolling_prospective_for_endpoint(b, rng_seed=20261123 + idx * 31)
    return b.endpoint_id, ra, rb, rc, anch, rr, round(time.time() - ts, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-Cohort ADNI and NACC Survival Evolution Pipeline")
    parser.add_argument("--adni-raw-dir", default=os.environ.get("ADNI_RAW_DIR", "data/raw/adni_route_a"))
    parser.add_argument("--adni-extract-dir", default=os.environ.get("ADNI_EXTRACT_DIR", "data/raw/adni_extract"))
    parser.add_argument("--nacc-parquet", default=os.environ.get("NACC_PARQUET", "data/raw/nacc/investigator_nacc75.parquet"))
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--pilot", action="store_true", help="Run strict audits + 10-rep pilot mode")
    parser.add_argument("--n-reps", type=int, default=200, help="Number of tie-breaking replications for Trajectory B")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes across endpoints")
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    n_reps = 10 if args.pilot else args.n_reps
    mode_tag = "pilot" if args.pilot else "full"
    print(f"=== Dual-Cohort Survival Pipeline ({mode_tag}, n_reps_B={n_reps}, workers={args.workers}) ===", flush=True)

    adni_bundles, adni_meta = build_adni_endpoints(args.adni_raw_dir, args.adni_extract_dir)
    nacc_bundles, nacc_meta = build_nacc_endpoints(args.nacc_parquet)
    all_bundles = adni_bundles + nacc_bundles
    print(f"Built {len(all_bundles)} endpoints in {time.time() - t0:.1f}s.", flush=True)

    print("\n--- Step 1: Strict Anti-Leakage & Lifelines Cross-Verification Audits ---", flush=True)
    audit_report = run_pilot_audits(all_bundles, ("cox_std", "cox_ridge_cv"))

    print("\n--- Step 2 & 3: Parallel Three-Trajectory (A/B/C) + Rolling Prospective ---", flush=True)
    all_a, all_b, all_c, all_roll = [], [], [], []
    anchors = {}
    tasks = [(idx, b, n_reps) for idx, b in enumerate(all_bundles)]
    n_workers = min(args.workers, len(tasks))
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for ep_id, ra, rb, rc, anch, rr, dt_s in ex.map(_worker_endpoint, tasks):
                all_a.extend(ra)
                all_b.extend(rb)
                all_c.extend(rc)
                all_roll.extend(rr)
                anchors[ep_id] = anch
                print(f"  Completed {ep_id:20s} in {dt_s:5.1f}s (A={len(ra)}, B={len(rb)}, C={len(rc)}, Roll={len(rr)})", flush=True)
    else:
        for tsk in tasks:
            ep_id, ra, rb, rc, anch, rr, dt_s = _worker_endpoint(tsk)
            all_a.extend(ra)
            all_b.extend(rb)
            all_c.extend(rc)
            all_roll.extend(rr)
            anchors[ep_id] = anch
            print(f"  Completed {ep_id:20s} in {dt_s:5.1f}s (A={len(ra)}, B={len(rb)}, C={len(rc)}, Roll={len(rr)})", flush=True)

    print("\n--- Step 4: ADNI Route A Real Historical Snapshot Fidelity Audit ---", flush=True)
    fidelity_rows = run_adni_real_snapshot_fidelity_audit(args.adni_raw_dir, args.adni_extract_dir, adni_bundles)
    print(f"  Completed real-snapshot fidelity audit ({len(fidelity_rows)} rows)", flush=True)

    # Write aggregate outputs
    sub_dir = os.path.join(args.out_dir, f"survival_{mode_tag}")
    os.makedirs(sub_dir, exist_ok=True)
    pd.DataFrame(all_a).to_csv(os.path.join(sub_dir, "trajectory_a_calendar_cutoff.csv"), index=False)
    pd.DataFrame(all_b).to_csv(os.path.join(sub_dir, "trajectory_b_chronological_doubling.csv"), index=False)
    pd.DataFrame(all_c).to_csv(os.path.join(sub_dir, "trajectory_c_locked_unfreezing.csv"), index=False)
    pd.DataFrame(all_roll).to_csv(os.path.join(sub_dir, "rolling_prospective.csv"), index=False)
    pd.DataFrame(fidelity_rows).to_csv(os.path.join(sub_dir, "adni_real_snapshot_fidelity.csv"), index=False)

    summary_meta = {
        "mode": mode_tag,
        "n_reps_b": n_reps,
        "elapsed_seconds": round(time.time() - t0, 2),
        "features": FEATURES_9,
        "adni_meta": adni_meta,
        "nacc_meta": nacc_meta,
        "pilot_audit": audit_report,
        "anchor_zero_drift": anchors,
    }
    with open(os.path.join(sub_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, indent=2)

    print(f"\n=== Survival Pipeline ({mode_tag}) FINISHED in {time.time() - t0:.1f}s -> {sub_dir} ===")


if __name__ == "__main__":
    main()
