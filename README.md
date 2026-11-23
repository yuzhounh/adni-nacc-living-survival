# ADNI + NACC Twenty-Year Living-Cohort Survival Evolution

Standalone, pure-Python analytical pipeline for **Twenty-year living-cohort evolution, administrative censoring maturation and rolling prospective transportability of Alzheimer's disease survival models in ADNI and NACC**.

## Overview

This repository evaluates eight paired Alzheimer's disease survival endpoints across enrolled ADNI (`ADNI1`--`ADNI4`, `N = 2,972`) and NACC Freeze 75 (`UDS 1.0`--`4.0`, 46 centers, `N = 57,272`) cohorts over 2005--2026:

1. **Three-trajectory causal decomposition** on deterministic 30% stratified evaluation reserves: annual calendar cutoffs (Trajectory A, 2008--2026), chronological sample-size doubling (Trajectory B), and locked `<= 2008` baseline cohort unfreezing with zero new recruits (Trajectory C).
2. **Rolling prospective transportability** (2009--2022) comparing expanding versus 5-year sliding windows on strictly future enrollees.
3. **Exact vectorized Efron Cox PH** versus 5-fold event-stratified Verweij--van Houwelingen CV Ridge-Cox, paired with exact O(N log N) Fenwick-tree Harrell's and Uno's IPCW C-indices, Graf IPCW Brier/IBS, calibration slopes, and fixed-SD standardized log(HR) drift.
4. **ADNI historical release fidelity audit** comparing genuinely archived `ADNIMERGE 2023-09` and `DXSUM 2025-07` releases against calendar-truncated newer releases.

## Repository Structure

- `survival_engine.py`: Exact Efron Cox solver, warm-started CV Ridge-Cox, Fenwick-tree C-indices, and IPCW Brier/calibration metrics.
- `build_cohorts.py`: Read-only ADNI and NACC cohort builder with nine cross-version harmonized covariates.
- `run_survival_evolution.py`: End-to-end CLI running strict pilot audits (`lifelines` cross-verification `< 1e-5`, zero-drift anchors) and full runs.
- `make_figures.py` / `make_tables.py`: Regenerate all manuscript figures and `booktabs` LaTeX tables from aggregate CSV outputs.
- `tests/test_survival_engine.py`: Six pure-synthetic unit tests (zero private data required).
- `data/README.md`: Instructions for obtaining ADNI and NACC files under approved DUAs.

## Quickstart

```bash
pip install -r requirements.txt

# Run pure-synthetic unit tests (no restricted data required)
python tests/test_survival_engine.py

# Run strict pilot audit against local ADNI/NACC data
python run_survival_evolution.py --pilot --adni-raw-dir <ADNI_RAW> --adni-extract-dir <ADNI_EXTRACT> --nacc-parquet <NACC_PARQUET> --out-dir outputs

# Run full analysis and regenerate figures/tables
python run_survival_evolution.py --n-reps 100 --workers 8 --out-dir outputs
python make_figures.py --res-dir outputs/survival_full --fig-dir outputs/figures
python make_tables.py --res-dir outputs/survival_full --tab-dir outputs/tables
```
