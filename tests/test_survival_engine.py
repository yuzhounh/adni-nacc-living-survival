"""Pure synthetic unit tests for the exact Efron Cox survival engine (zero private data required)."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from survival_engine import (
    baseline_cumhaz_at,
    compute_eval_censoring_km,
    eval_step_surv,
    evaluate_survival_model,
    fit_cox_model,
    fit_efron_cox_std_x,
    ipcw_brier_score,
    km_survival_curve,
    weighted_c_index,
)


def _naive_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray, weights=None, tau=None) -> float:
    n = len(time)
    w = np.ones(n) if weights is None else weights
    num = 0.0
    den = 0.0
    for i in range(n):
        if event[i] != 1:
            continue
        if tau is not None and time[i] > tau:
            continue
        for j in range(n):
            if i == j:
                continue
            if (time[i] < time[j]) or (time[i] == time[j] and event[j] == 0):
                den += w[i]
                if risk[i] > risk[j]:
                    num += w[i]
                elif risk[i] == risk[j]:
                    num += 0.5 * w[i]
    return num / den if den > 0 else np.nan


def test_fenwick_c_index_exact_with_ties():
    rng = np.random.default_rng(42)
    n = 160
    # Deliberately create tied times and tied risk scores
    t = rng.choice([1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 24.0, 36.0, 48.0, 60.0], size=n)
    e = rng.binomial(1, 0.45, size=n)
    r = np.round(rng.normal(0, 1, size=n), 1)
    w = rng.uniform(0.5, 4.0, size=n)

    c_fast = weighted_c_index(t, e, r)
    c_slow = _naive_c_index(t, e, r)
    assert abs(c_fast - c_slow) < 1e-13, f"Harrell mismatch: {c_fast} vs {c_slow}"

    u_fast = weighted_c_index(t, e, r, weights=w, tau=36.0)
    u_slow = _naive_c_index(t, e, r, weights=w, tau=36.0)
    assert abs(u_fast - u_slow) < 1e-13, f"Uno mismatch: {u_fast} vs {u_slow}"


def test_efron_cox_matches_lifelines():
    rng = np.random.default_rng(123)
    n, p = 600, 6
    X = rng.normal(0, 1, size=(n, p))
    true_beta = np.array([0.6, -0.4, 0.3, 0.0, -0.25, 0.45])
    lin = X @ true_beta
    t_ev = rng.exponential(scale=30.0 * np.exp(-lin))
    t_cens = rng.uniform(10.0, 70.0, size=n)
    time = np.round(np.minimum(t_ev, t_cens), 1) + 0.1  # includes ties
    event = (t_ev <= t_cens).astype(int)

    # Standardize X directly
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    lam = 0.01
    beta_ours, conv, _, ev_t, ch0 = fit_efron_cox_std_x(Xs, time, event, lambda_pen=lam)
    assert conv

    df = pd.DataFrame(Xs, columns=[f"x{j}" for j in range(p)])
    df["T"] = time
    df["E"] = event
    # lifelines minimizes -log_lik / N + 0.5 * penalizer * ||beta||^2 => penalizer = lam / N
    cph = CoxPHFitter(penalizer=lam / n, l1_ratio=0.0)
    cph.fit(df, duration_col="T", event_col="E")
    beta_ll = cph.params_.values

    max_diff = float(np.max(np.abs(beta_ours - beta_ll)))
    assert max_diff < 1e-5, f"Cox beta diff vs lifelines too large: {max_diff}"

    # Verify baseline cumulative hazard at t=36 (note lifelines centers X internally, so compare H0 * exp(0) on centered X)
    m = fit_cox_model(Xs, time, event, model_type="cox_std", lambda_std=lam)
    h0_36 = baseline_cumhaz_at(m, 36.0)
    bh_df = cph.baseline_cumulative_hazard_
    h0_ll = float(bh_df.loc[bh_df.index <= 36.0].iloc[-1, 0])
    assert abs(h0_36 - h0_ll) < 1e-4, f"H0(36) mismatch: {h0_36} vs {h0_ll}"


def test_ridge_cv_and_evaluation_pipeline():
    rng = np.random.default_rng(999)
    n_tr, n_ev, p = 240, 200, 9
    X_tr = rng.normal(0, 1, size=(n_tr, p))
    X_ev = rng.normal(0, 1, size=(n_ev, p))
    # Inject missing values to test leakage-free train-only imputation
    X_tr[rng.random(X_tr.shape) < 0.08] = np.nan
    X_ev[rng.random(X_ev.shape) < 0.08] = np.nan

    tb = np.array([0.5, -0.3, 0.2, 0.4, 0.6, -0.2, 0.1, 0.3, -0.35])
    t_tr = np.minimum(rng.exponential(35.0 * np.exp(-np.nan_to_num(X_tr) @ tb)), 72.0) + 0.5
    e_tr = (t_tr < 55.0).astype(int)
    t_ev = np.minimum(rng.exponential(35.0 * np.exp(-np.nan_to_num(X_ev) @ tb)), 72.0) + 0.5
    e_ev = (t_ev < 55.0).astype(int)

    m_std = fit_cox_model(X_tr, t_tr, e_tr, model_type="cox_std")
    m_ridge = fit_cox_model(X_tr, t_tr, e_tr, model_type="cox_ridge_cv")
    assert m_std.converged and m_ridge.converged

    g_ut, g_s = compute_eval_censoring_km(t_ev, e_ev)
    km_ut, km_s = km_survival_curve(t_ev, e_ev)
    ref_sd = m_std.sd_x
    res_std = evaluate_survival_model(m_std, X_ev, t_ev, e_ev, g_ut, g_s, km_ut, km_s, ref_sd, m_std.beta)
    res_ridge = evaluate_survival_model(m_ridge, X_ev, t_ev, e_ev, g_ut, g_s, km_ut, km_s, ref_sd, m_std.beta)

    assert 0.5 < res_std["c_harrell"] < 1.0
    assert 0.5 < res_std["c_uno_48m"] < 1.0
    assert 0.0 < res_std["brier_36m"] < 0.35
    assert 0.0 < res_std["ibs_12_60m"] < 0.35
    assert res_std["coef_l2_to_ref"] < 1e-10
    assert abs(res_std["coef_cos_to_ref"] - 1.0) < 1e-10
    assert np.isfinite(res_ridge["cal_slope"])


def test_ipcw_brier_degenerate_perfect_and_null():
    t = np.array([10.0, 20.0, 40.0, 50.0, 60.0, 15.0])
    e = np.array([1, 1, 0, 1, 0, 1])
    g_ut, g_s = compute_eval_censoring_km(t, e)
    p_perf = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    bs_perf = ipcw_brier_score(t, e, p_perf, 36.0, g_ut, g_s)
    assert bs_perf < 1e-12
    p_half = np.full(len(t), 0.5)
    bs_half = ipcw_brier_score(t, e, p_half, 36.0, g_ut, g_s)
    assert abs(bs_half - 0.25) < 0.05


def test_low_event_graceful_handling():
    X = np.random.default_rng(7).normal(0, 1, size=(20, 4))
    t = np.linspace(1.0, 20.0, 20)
    e = np.zeros(20, dtype=int)
    e[5] = 1  # only 1 event
    m = fit_cox_model(X, t, e, model_type="cox_std")
    assert not m.converged
    assert np.all(m.beta == 0.0)


def test_km_curve_known_values():
    t = np.array([1.0, 2.0, 2.0, 3.0, 4.0])
    e = np.array([1, 1, 0, 1, 0])
    ut, s = km_survival_curve(t, e)
    # t=1: 4/5 = 0.8; t=2: 0.8 * 3/4 = 0.6; t=3: 0.6 * 1/2 = 0.3
    assert np.allclose(ut, [1.0, 2.0, 3.0])
    assert np.allclose(s, [0.8, 0.6, 0.3])
    assert np.allclose(eval_step_surv(ut, s, np.array([0.5, 1.0, 2.0, 3.5]), left_limit=False), [1.0, 0.8, 0.6, 0.3])
    assert np.allclose(eval_step_surv(ut, s, np.array([1.0, 2.0, 3.0]), left_limit=True), [1.0, 0.8, 0.6])


def run_all() -> None:
    tests = [
        test_fenwick_c_index_exact_with_ties,
        test_efron_cox_matches_lifelines,
        test_ridge_cv_and_evaluation_pipeline,
        test_ipcw_brier_degenerate_perfect_and_null,
        test_low_event_graceful_handling,
        test_km_curve_known_values,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"All {len(tests)}/{len(tests)} synthetic unit tests passed.")


if __name__ == "__main__":
    run_all()
