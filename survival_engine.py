"""Fast vectorized exact Efron Cox PH (warm-started, upper-triangle, sparse-event), Ridge-Cox CV, and O(N log N) IPCW engine for ADNI/NACC survival evolution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import numpy as np


RIDGE_LAMBDA_GRID = (200.0, 50.0, 10.0, 2.0, 0.5, 0.1, 0.01)


@dataclass
class CoxModelResult:
    beta: np.ndarray
    lambda_pen: float
    converged: bool
    n_iter: int
    event_times: np.ndarray
    cum_base_haz: np.ndarray
    max_train_time: float
    mean_x: np.ndarray
    sd_x: np.ndarray
    medians: np.ndarray


def fit_train_preprocessor(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit median imputation and z-score scaling strictly inside the training set."""
    X = np.asarray(X_train, dtype=np.float64)
    p = X.shape[1]
    medians = np.zeros(p, dtype=np.float64)
    X_imp = X.copy()
    for j in range(p):
        col = X[:, j]
        valid = np.isfinite(col)
        med = float(np.median(col[valid])) if np.any(valid) else 0.0
        medians[j] = med
        if not np.all(valid):
            X_imp[~valid, j] = med
    mean_x = X_imp.mean(axis=0)
    sd_x = X_imp.std(axis=0, ddof=0)
    sd_safe = np.where(sd_x > 1e-12, sd_x, 1.0)
    X_std = (X_imp - mean_x) / sd_safe
    return X_std, medians, mean_x, sd_safe


def transform_with_preprocessor(
    X: np.ndarray, medians: np.ndarray, mean_x: np.ndarray, sd_safe: np.ndarray
) -> np.ndarray:
    """Apply training-derived median imputation and z-score scaling without leakage."""
    X_imp = np.asarray(X, dtype=np.float64).copy()
    for j in range(X_imp.shape[1]):
        bad = ~np.isfinite(X_imp[:, j])
        if np.any(bad):
            X_imp[bad, j] = medians[j]
    return (X_imp - mean_x) / sd_safe


class _EfronCache:
    """Precomputed descending-time structure enabling 10x faster exact Efron Newton-Raphson & warm-started CV."""

    def __init__(self, X: np.ndarray, time: np.ndarray, event: np.ndarray):
        self.n, self.p = X.shape
        order = np.argsort(-time, kind="mergesort")
        self.Xs = np.ascontiguousarray(X[order], dtype=np.float64)
        ts = time[order]
        es = event[order].astype(np.int64)
        self.n_events = int(np.sum(es))
        if self.n_events == 0:
            return

        change = np.empty(self.n, dtype=bool)
        change[0] = True
        change[1:] = ts[1:] != ts[:-1]
        block_starts = np.flatnonzero(change)
        block_ends = np.empty_like(block_starts)
        block_ends[:-1] = block_starts[1:] - 1
        block_ends[-1] = self.n - 1

        ev_prefix = np.cumsum(es)
        prev_ev = np.zeros_like(block_starts)
        prev_ev[1:] = ev_prefix[block_starts[1:] - 1]
        d_block = ev_prefix[block_ends] - prev_ev

        ev_mask = d_block > 0
        self.ev_ends = block_ends[ev_mask]
        self.d_ev = d_block[ev_mask].astype(np.int64)
        self.n_ev_blocks = len(self.d_ev)
        self.ev_times_desc = ts[self.ev_ends]

        # Sparse event-only rows & their event-block membership for O(n_events) reduceat
        self.ev_rows = np.flatnonzero(es == 1)
        self.Xs_ev = np.ascontiguousarray(self.Xs[self.ev_rows])
        self.x_ev_sum = self.Xs_ev.sum(axis=0)
        # Starts within ev_rows for each event block (since ev_rows are strictly grouped by block!)
        self.ev_block_reduce_starts = np.empty(self.n_ev_blocks, dtype=np.int64)
        self.ev_block_reduce_starts[0] = 0
        if self.n_ev_blocks > 1:
            self.ev_block_reduce_starts[1:] = np.cumsum(self.d_ev[:-1])

        # Upper-triangle outer product indices (cuts Hessian memory/bandwidth in half)
        self.tri_i, self.tri_j = np.triu_indices(self.p)
        self.n_tri = len(self.tri_i)
        self.XX_tri = np.ascontiguousarray(self.Xs[:, self.tri_i] * self.Xs[:, self.tri_j])
        self.XX_ev_tri = np.ascontiguousarray(self.XX_tri[self.ev_rows])

        # Flat Efron tie expansion across all n_events terms
        self.block_ids = np.repeat(np.arange(self.n_ev_blocks, dtype=np.int64), self.d_ev)
        cum_d = np.cumsum(self.d_ev)
        starts_exp = np.empty_like(cum_d)
        starts_exp[0] = 0
        starts_exp[1:] = cum_d[:-1]
        within = np.arange(self.n_events, dtype=np.int64) - np.repeat(starts_exp, self.d_ev)
        self.fracs = within / np.repeat(self.d_ev.astype(np.float64), self.d_ev)
        self.has_ties = bool(np.any(self.d_ev > 1))

        # Compress risk-set prefix sums by partitioning observations into segments ending at ev_ends:
        # every observation before/at ev_ends[b] contributes to segment b!
        self.seg_starts = np.empty(self.n_ev_blocks, dtype=np.int64)
        self.seg_starts[0] = 0
        if self.n_ev_blocks > 1:
            self.seg_starts[1:] = self.ev_ends[:-1] + 1
        self.last_valid_row = int(self.ev_ends[-1]) + 1  # rows after last event never enter any risk set!


def efron_loglik_cached(cache: _EfronCache, beta: np.ndarray) -> float:
    """Fast exact unpenalized Efron partial log-likelihood using precomputed cache."""
    if cache.n_events == 0:
        return 0.0
    m = cache.last_valid_row
    eta = np.clip(cache.Xs[:m] @ beta, -50.0, 50.0)
    shift = float(np.max(eta))
    w = np.exp(eta - shift)
    seg_w = np.add.reduceat(w, cache.seg_starts)
    S0 = np.cumsum(seg_w)

    if cache.has_ties:
        w_ev = w[cache.ev_rows]
        D0 = np.add.reduceat(w_ev, cache.ev_block_reduce_starts)
        eff_S0 = np.maximum(S0[cache.block_ids] - cache.fracs * D0[cache.block_ids], 1e-300)
        return float(cache.x_ev_sum @ beta - np.sum(np.log(eff_S0) + shift))
    else:
        return float(cache.x_ev_sum @ beta - np.sum(np.log(np.maximum(S0, 1e-300)) + shift))


def efron_loglik(X: np.ndarray, time: np.ndarray, event: np.ndarray, beta: np.ndarray) -> float:
    return efron_loglik_cached(_EfronCache(X, time, event), beta)


def fit_efron_cox_cached(
    cache: _EfronCache,
    lambda_pen: float = 0.01,
    beta_init: Optional[np.ndarray] = None,
    max_iter: int = 50,
    tol: float = 1e-9,
    compute_baseline_haz: bool = True,
) -> Tuple[np.ndarray, bool, int, np.ndarray, np.ndarray]:
    """Fit L2-penalized exact Efron Cox PH via warm-started damped Newton-Raphson using upper-triangle reduceat."""
    p = cache.p
    beta = np.zeros(p, dtype=np.float64) if beta_init is None else beta_init.astype(np.float64).copy()
    if cache.n_events < 2:
        return beta, False, 0, np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    m = cache.last_valid_row
    Xs_m = cache.Xs[:m]
    XX_m = cache.XX_tri[:m]
    eye_p = np.eye(p, dtype=np.float64)
    tri_i, tri_j = cache.tri_i, cache.tri_j

    def eval_obj_grad_hess(b: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        eta = np.clip(Xs_m @ b, -40.0, 40.0)
        shift = float(np.max(eta))
        w = np.exp(eta - shift)

        # Segment-wise reduceat then cumsum over only n_ev_blocks (tiny compared to N!)
        S0 = np.cumsum(np.add.reduceat(w, cache.seg_starts))
        S1 = np.cumsum(np.add.reduceat(Xs_m * w[:, None], cache.seg_starts, axis=0), axis=0)
        S2_tri = np.cumsum(np.add.reduceat(XX_m * w[:, None], cache.seg_starts, axis=0), axis=0)

        if cache.has_ties:
            w_ev = w[cache.ev_rows]
            D0 = np.add.reduceat(w_ev, cache.ev_block_reduce_starts)
            D1 = np.add.reduceat(cache.Xs_ev * w_ev[:, None], cache.ev_block_reduce_starts, axis=0)
            D2_tri = np.add.reduceat(cache.XX_ev_tri * w_ev[:, None], cache.ev_block_reduce_starts, axis=0)

            bids = cache.block_ids
            fr = cache.fracs
            eff_S0 = np.maximum(S0[bids] - fr * D0[bids], 1e-300)
            eff_S1 = S1[bids] - fr[:, None] * D1[bids]
            eff_S2_tri = S2_tri[bids] - fr[:, None] * D2_tri[bids]
        else:
            eff_S0 = np.maximum(S0, 1e-300)
            eff_S1 = S1
            eff_S2_tri = S2_tri

        inv_S0 = 1.0 / eff_S0
        obj = float(cache.x_ev_sum @ b - np.sum(np.log(eff_S0) + shift) - 0.5 * lambda_pen * (b @ b))
        mean_r = eff_S1 * inv_S0[:, None]
        grad = cache.x_ev_sum - mean_r.sum(axis=0) - lambda_pen * b

        # Sum of upper-triangle second moments: (eff_S2_tri / eff_S0).sum(axis=0)
        s2_sum_tri = inv_S0 @ eff_S2_tri  # shape (n_tri,) via BLAS gemv!
        s2_mat = np.zeros((p, p), dtype=np.float64)
        s2_mat[tri_i, tri_j] = s2_sum_tri
        s2_mat[tri_j, tri_i] = s2_sum_tri

        info = s2_mat - (mean_r.T @ mean_r) + lambda_pen * eye_p
        return obj, grad, info

    converged = False
    obj, grad, info = eval_obj_grad_hess(beta)
    n_iter = 0
    for it in range(1, max_iter + 1):
        n_iter = it
        try:
            step = np.linalg.solve(info + 1e-10 * eye_p, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(info + 1e-8 * eye_p, grad, rcond=None)[0]

        max_step = float(np.max(np.abs(step)))
        if max_step > 5.0:
            step = step * (5.0 / max_step)

        alpha = 1.0
        accepted = False
        for _ in range(25):
            cand = beta + alpha * step
            cand_obj, cand_grad, cand_info = eval_obj_grad_hess(cand)
            if np.isfinite(cand_obj) and cand_obj >= obj - 1e-12:
                beta, obj, grad, info = cand, cand_obj, cand_grad, cand_info
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            break
        if float(np.max(np.abs(alpha * step))) < tol and float(np.max(np.abs(grad))) < 1e-6:
            converged = True
            break

    if not compute_baseline_haz:
        return beta, converged, n_iter, np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    eta = np.clip(Xs_m @ beta, -40.0, 40.0)
    shift = float(np.max(eta))
    w = np.exp(eta - shift)
    S0_unshifted = np.cumsum(np.add.reduceat(w, cache.seg_starts)) * np.exp(shift)
    dh0_desc = cache.d_ev.astype(np.float64) / S0_unshifted
    ev_times_asc = cache.ev_times_desc[::-1].copy()
    cum_h0_asc = np.cumsum(dh0_desc[::-1])
    return beta, converged, n_iter, ev_times_asc, cum_h0_asc


def fit_efron_cox_std_x(
    X_std: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    lambda_pen: float = 0.01,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, bool, int, np.ndarray, np.ndarray]:
    cache = _EfronCache(X_std, time, event)
    return fit_efron_cox_cached(cache, lambda_pen=lambda_pen, max_iter=max_iter, tol=tol, compute_baseline_haz=True)


def fit_cox_model(
    X_train: np.ndarray,
    time_train: np.ndarray,
    event_train: np.ndarray,
    model_type: str = "cox_std",
    lambda_std: float = 0.01,
    lambda_grid: Sequence[float] = RIDGE_LAMBDA_GRID,
    cv_folds: int = 5,
    cv_seed: int = 20261123,
) -> CoxModelResult:
    """Fit standardized exact Efron Cox (`cox_std`) or warm-started 5-fold stratified Verweij-van Houwelingen Ridge-Cox."""
    X_std, medians, mean_x, sd_safe = fit_train_preprocessor(X_train)
    t = np.asarray(time_train, dtype=np.float64)
    e = np.asarray(event_train, dtype=np.int64)
    max_t = float(np.max(t)) if len(t) > 0 else 0.0

    if int(np.sum(e)) < 2:
        return CoxModelResult(
            beta=np.zeros(X_std.shape[1], dtype=np.float64),
            lambda_pen=float(lambda_std),
            converged=False,
            n_iter=0,
            event_times=np.array([], dtype=np.float64),
            cum_base_haz=np.array([], dtype=np.float64),
            max_train_time=max_t,
            mean_x=mean_x,
            sd_x=sd_safe,
            medians=medians,
        )

    full_cache = _EfronCache(X_std, t, e)

    if model_type == "cox_std":
        chosen_lam = float(lambda_std)
        beta, converged, n_iter, ev_times, cum_h0 = fit_efron_cox_cached(
            full_cache, lambda_pen=chosen_lam, compute_baseline_haz=True
        )
    elif model_type == "cox_ridge_cv":
        n_ev = int(np.sum(e))
        k_eff = min(cv_folds, max(2, n_ev))
        lam_desc = sorted([float(x) for x in lambda_grid], reverse=True)
        if n_ev < 5:
            chosen_lam = lam_desc[len(lam_desc) // 2]
        else:
            rng = np.random.default_rng(cv_seed)
            folds = np.zeros(len(t), dtype=np.int64)
            for cls in (0, 1):
                idx = np.flatnonzero(e == cls)
                perm = rng.permutation(idx)
                for i_pos, r_idx in enumerate(perm):
                    folds[r_idx] = i_pos % k_eff

            # Pre-cache each training fold ONCE across all lambdas, and warm-start along descending lambda path
            fold_caches = []
            for f in range(k_eff):
                tr_mask = folds != f
                if int(np.sum(e[tr_mask])) >= 2:
                    fold_caches.append(_EfronCache(X_std[tr_mask], t[tr_mask], e[tr_mask]))
                else:
                    fold_caches.append(None)

            cv_scores = np.zeros(len(lam_desc), dtype=np.float64)
            cv_counts = np.zeros(len(lam_desc), dtype=np.int64)
            for fc in fold_caches:
                if fc is None:
                    continue
                b_prev = None
                for li, lam in enumerate(lam_desc):
                    b_f, _, _, _, _ = fit_efron_cox_cached(
                        fc, lambda_pen=lam, beta_init=b_prev, compute_baseline_haz=False
                    )
                    if np.all(np.isfinite(b_f)):
                        b_prev = b_f
                        ll_full = efron_loglik_cached(full_cache, b_f)
                        ll_tr = efron_loglik_cached(fc, b_f)
                        cv_scores[li] += (ll_full - ll_tr)
                        cv_counts[li] += 1

            valid_lams = cv_counts == k_eff
            if np.any(valid_lams):
                best_idx = int(np.argmax(np.where(valid_lams, cv_scores, -np.inf)))
                chosen_lam = lam_desc[best_idx]
            else:
                chosen_lam = lam_desc[-1]

        beta, converged, n_iter, ev_times, cum_h0 = fit_efron_cox_cached(
            full_cache, lambda_pen=chosen_lam, compute_baseline_haz=True
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return CoxModelResult(
        beta=beta,
        lambda_pen=chosen_lam,
        converged=converged,
        n_iter=n_iter,
        event_times=ev_times,
        cum_base_haz=cum_h0,
        max_train_time=max_t,
        mean_x=mean_x,
        sd_x=sd_safe,
        medians=medians,
    )


def baseline_cumhaz_at(model: CoxModelResult, tau: float) -> float:
    """Right-continuous Breslow baseline cumulative hazard H0(tau)."""
    if len(model.event_times) == 0:
        return 0.0
    idx = int(np.searchsorted(model.event_times, tau, side="right")) - 1
    if idx < 0:
        return 0.0
    return float(model.cum_base_haz[idx])


def predict_risk(model: CoxModelResult, X_eval: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (linear_predictor, predicted_event_prob_by_tau) on X_eval."""
    X_std = transform_with_preprocessor(X_eval, model.medians, model.mean_x, model.sd_x)
    eta = np.clip(X_std @ model.beta, -30.0, 30.0)
    h0 = baseline_cumhaz_at(model, tau)
    surv = np.exp(-h0 * np.exp(eta))
    risk = np.clip(1.0 - surv, 0.0, 1.0)
    return eta, risk


def km_survival_curve(time: np.ndarray, event: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Exact Kaplan-Meier curve returning ascending unique event times and survival probabilities."""
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=np.int64)
    order = np.argsort(t, kind="mergesort")
    ts = t[order]
    es = e[order]
    uniq_t, idx_first, counts = np.unique(ts, return_index=True, return_counts=True)
    n = len(ts)
    at_risk = n - idx_first
    ev_cumsum = np.cumsum(es)
    ev_end = idx_first + counts - 1
    prev = np.zeros_like(idx_first)
    prev[1:] = ev_cumsum[idx_first[1:] - 1]
    d = ev_cumsum[ev_end] - prev

    mask = d > 0
    ut = uniq_t[mask]
    surv = np.cumprod(1.0 - d[mask] / at_risk[mask])
    return ut, surv


def eval_step_surv(ut: np.ndarray, surv: np.ndarray, t_query: np.ndarray, left_limit: bool = False) -> np.ndarray:
    """Evaluate step survival function S(t) or left-limit S(t-) at query points."""
    q = np.asarray(t_query, dtype=np.float64)
    if len(ut) == 0:
        return np.ones_like(q, dtype=np.float64)
    side = "left" if left_limit else "right"
    idx = np.searchsorted(ut, q, side=side) - 1
    out = np.ones_like(q, dtype=np.float64)
    valid = idx >= 0
    out[valid] = surv[idx[valid]]
    return out


class _FenwickTree:
    """1-indexed Fenwick tree supporting weighted prefix sums in O(log M)."""

    def __init__(self, size: int):
        self.size = size
        self.tree = np.zeros(size + 1, dtype=np.float64)

    def add(self, idx: int, val: float) -> None:
        i = idx
        n = self.size
        t = self.tree
        while i <= n:
            t[i] += val
            i += i & -i

    def query(self, idx: int) -> float:
        s = 0.0
        i = idx
        t = self.tree
        while i > 0:
            s += t[i]
            i -= i & -i
        return s


def weighted_c_index(
    time: np.ndarray,
    event: np.ndarray,
    risk_score: np.ndarray,
    weights: Optional[np.ndarray] = None,
    tau: Optional[float] = None,
) -> float:
    """Exact O(N log N) Harrell's (weights=None, tau=None) or Uno's IPCW C-index via Fenwick tree."""
    t = np.asarray(time, dtype=np.float64)
    e = np.asarray(event, dtype=np.int64)
    r = np.asarray(risk_score, dtype=np.float64)
    n = len(t)
    if n == 0 or int(np.sum(e)) == 0:
        return float("nan")
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)

    _, inv = np.unique(r, return_inverse=True)
    ranks = inv + 1
    m = int(np.max(ranks))

    order = np.lexsort((e, -t))
    ts = t[order]
    es = e[order]
    rk = ranks[order]
    ws = w[order]

    ft_cnt = _FenwickTree(m)
    ft_tie = np.zeros(m + 1, dtype=np.float64)
    total_inserted = 0.0

    num = 0.0
    den = 0.0
    i = 0
    while i < n:
        cur_t = ts[i]
        j = i
        while j < n and ts[j] == cur_t:
            j += 1

        k = i
        while k < j and es[k] == 0:
            r_k = int(rk[k])
            ft_cnt.add(r_k, 1.0)
            ft_tie[r_k] += 1.0
            total_inserted += 1.0
            k += 1

        ev_start = k
        if ev_start < j and (tau is None or cur_t <= tau):
            for idx_e in range(ev_start, j):
                wi = float(ws[idx_e])
                if wi <= 0.0 or not np.isfinite(wi):
                    continue
                r_e = int(rk[idx_e])
                n_smaller_r = ft_cnt.query(r_e - 1) if r_e > 1 else 0.0
                n_equal_r = float(ft_tie[r_e])
                num += wi * (n_smaller_r + 0.5 * n_equal_r)
                den += wi * total_inserted

        for idx_e in range(ev_start, j):
            r_e = int(rk[idx_e])
            ft_cnt.add(r_e, 1.0)
            ft_tie[r_e] += 1.0
            total_inserted += 1.0

        i = j

    return float(num / den) if den > 0.0 else float("nan")


def compute_eval_censoring_km(time_eval: np.ndarray, event_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reverse Kaplan-Meier censoring survival function G_eval(t) on the evaluation set."""
    return km_survival_curve(time_eval, 1 - np.asarray(event_eval, dtype=np.int64))


def ipcw_brier_score(
    time_eval: np.ndarray,
    event_eval: np.ndarray,
    pred_risk: np.ndarray,
    tau: float,
    g_ut: np.ndarray,
    g_surv: np.ndarray,
    g_floor: float = 0.02,
) -> float:
    """Graf et al. (1999) IPCW Brier score at horizon tau."""
    t = np.asarray(time_eval, dtype=np.float64)
    e = np.asarray(event_eval, dtype=np.int64)
    p = np.asarray(pred_risk, dtype=np.float64)
    g_t_minus = np.maximum(eval_step_surv(g_ut, g_surv, t, left_limit=True), g_floor)
    g_tau = max(float(eval_step_surv(g_ut, g_surv, np.array([tau]), left_limit=False)[0]), g_floor)

    is_case = (t <= tau) & (e == 1)
    is_ctrl = t > tau
    terms = np.zeros_like(t, dtype=np.float64)
    terms[is_case] = ((1.0 - p[is_case]) ** 2) / g_t_minus[is_case]
    terms[is_ctrl] = (p[is_ctrl] ** 2) / g_tau
    return float(np.mean(terms))


def evaluate_survival_model(
    model: CoxModelResult,
    X_eval: np.ndarray,
    time_eval: np.ndarray,
    event_eval: np.ndarray,
    g_ut: np.ndarray,
    g_surv: np.ndarray,
    km_ut: np.ndarray,
    km_surv: np.ndarray,
    ref_sd: np.ndarray,
    ref_beta_std: Optional[np.ndarray] = None,
    uno_tau: float = 48.0,
    ibs_grid: Sequence[float] = (12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0),
) -> Dict[str, float]:
    """Compute complete discrimination, IPCW Brier/IBS, absolute calibration, slope, and HR stability metrics."""
    t_ev = np.asarray(time_eval, dtype=np.float64)
    e_ev = np.asarray(event_eval, dtype=np.int64)

    eta_eval, risk_36 = predict_risk(model, X_eval, 36.0)
    _, risk_60 = predict_risk(model, X_eval, 60.0)

    c_harrell = weighted_c_index(t_ev, e_ev, eta_eval, weights=None, tau=None)

    g_t_minus = np.maximum(eval_step_surv(g_ut, g_surv, t_ev, left_limit=True), 0.02)
    uno_weights = 1.0 / (g_t_minus ** 2)
    c_uno = weighted_c_index(t_ev, e_ev, eta_eval, weights=uno_weights, tau=uno_tau)

    brier_36 = ipcw_brier_score(t_ev, e_ev, risk_36, 36.0, g_ut, g_surv)
    brier_60 = ipcw_brier_score(t_ev, e_ev, risk_60, 60.0, g_ut, g_surv)
    briers = []
    for tau in ibs_grid:
        _, r_tau = predict_risk(model, X_eval, float(tau))
        briers.append(ipcw_brier_score(t_ev, e_ev, r_tau, float(tau), g_ut, g_surv))
    ibs = float(np.trapezoid(briers, ibs_grid) / (ibs_grid[-1] - ibs_grid[0]))

    obs_36 = float(1.0 - eval_step_surv(km_ut, km_surv, np.array([36.0]), left_limit=False)[0])
    obs_60 = float(1.0 - eval_step_surv(km_ut, km_surv, np.array([60.0]), left_limit=False)[0])
    pred_36 = float(np.mean(risk_36))
    pred_60 = float(np.mean(risk_60))
    oe_36 = float(pred_36 / obs_36) if obs_36 > 1e-8 else float("nan")
    oe_60 = float(pred_60 / obs_60) if obs_60 > 1e-8 else float("nan")
    bias_36_pct = float((pred_36 - obs_36) * 100.0)
    bias_60_pct = float((pred_60 - obs_60) * 100.0)

    if float(np.std(eta_eval)) > 1e-9 and int(np.sum(e_ev)) >= 5:
        slope_b, slope_conv, _, _, _ = fit_efron_cox_std_x(
            eta_eval[:, None], t_ev, e_ev, lambda_pen=1e-8, max_iter=40
        )
        cal_slope = float(slope_b[0]) if slope_conv and np.isfinite(slope_b[0]) else float("nan")
    else:
        cal_slope = float("nan")

    beta_ref_scale = model.beta * (ref_sd / model.sd_x)
    if ref_beta_std is not None:
        diff = beta_ref_scale - ref_beta_std
        coef_l2 = float(np.linalg.norm(diff))
        n1 = float(np.linalg.norm(beta_ref_scale))
        n2 = float(np.linalg.norm(ref_beta_std))
        coef_cos = float((beta_ref_scale @ ref_beta_std) / (n1 * n2)) if (n1 > 1e-12 and n2 > 1e-12) else float("nan")
    else:
        coef_l2 = float("nan")
        coef_cos = float("nan")

    out = {
        "c_harrell": c_harrell,
        "c_uno_48m": c_uno,
        "brier_36m": brier_36,
        "brier_60m": brier_60,
        "ibs_12_60m": ibs,
        "pred_risk_36m": pred_36,
        "obs_risk_36m": obs_36,
        "oe_ratio_36m": oe_36,
        "risk_bias_36m_pct": bias_36_pct,
        "pred_risk_60m": pred_60,
        "obs_risk_60m": obs_60,
        "oe_ratio_60m": oe_60,
        "risk_bias_60m_pct": bias_60_pct,
        "cal_slope": cal_slope,
        "coef_l2_to_ref": coef_l2,
        "coef_cos_to_ref": coef_cos,
        "lambda_pen": float(model.lambda_pen),
        "converged": int(model.converged),
        "max_train_fu_m": float(model.max_train_time),
        "horizon_36m_reached": int(model.max_train_time >= 36.0),
        "horizon_60m_reached": int(model.max_train_time >= 60.0),
    }
    for j, bval in enumerate(beta_ref_scale):
        out[f"beta_ref_{j}"] = float(bval)
    return out
