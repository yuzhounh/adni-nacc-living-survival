"""Read-only dual-cohort builder (ADNI Route A + NACC Freeze 75) for dual-cohort ADNI/NACC survival analysis."""
from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


FEATURES_9 = [
    "age",
    "sex_male",
    "educ",
    "apoe4_count",
    "cdrsb",
    "faq",
    "traila",
    "trailb",
    "animals",
]

ADNI_PHASES = ("ADNI1", "ADNIGO", "ADNI2", "ADNI3", "ADNI4")
FAQ_ITEMS_NACC = [
    "BILLS", "TAXES", "SHOPPING", "GAMES", "STOVE",
    "MEALPREP", "EVENTS", "PAYATTN", "REMDATES", "TRAVEL",
]


@dataclass
class EndpointBundle:
    cohort: str
    endpoint_id: str
    label: str
    baseline_df: pd.DataFrame  # columns: subj_id, bl_dt, stratum, phase_ver, bl_year, is_eval, FEATURES_9
    long_df: pd.DataFrame      # columns: subj_id, visit_dt, is_event
    final_cutoff: pd.Timestamp


def _clean_range(s: pd.Series, lo: float, hi: float) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce")
    return v.where((v >= lo) & (v <= hi), np.nan)


def _parse_apoe4_count(s: pd.Series) -> pd.Series:
    def _one(val):
        if pd.isna(val):
            return np.nan
        txt = str(val).strip()
        if txt in ("", "nan", "None"):
            return np.nan
        return float(txt.count("4"))
    return s.map(_one)


def _deterministic_stratified_holdout(
    subj_ids: pd.Series, strata: pd.Series, seed_tag: str, frac: float = 0.30
) -> np.ndarray:
    """Assign 30% stratified holdout deterministically using SHA-256 hashing of (seed_tag, subj_id)."""
    df = pd.DataFrame({"id": subj_ids.astype(str).values, "stratum": strata.astype(str).values})
    df["_hash"] = [
        hashlib.sha256(f"{seed_tag}|{sid}".encode("utf-8")).hexdigest()
        for sid in df["id"]
    ]
    is_eval = np.zeros(len(df), dtype=bool)
    for _, grp in df.groupby("stratum", sort=True):
        order = grp.sort_values(["_hash", "id"]).index.values
        n_eval = int(np.floor(len(order) * frac + 0.5))
        if len(order) >= 2 and n_eval == 0:
            n_eval = 1
        is_eval[order[:n_eval]] = True
    return is_eval


def compute_survival_at_cutoff(
    baseline_df: pd.DataFrame,
    long_df: pd.DataFrame,
    cutoff_dt: pd.Timestamp,
    allowed_ids: Optional[set] = None,
) -> pd.DataFrame:
    """Strictly reconstruct (time_months, event) using ONLY bl_dt <= cutoff_dt and visit_dt <= cutoff_dt."""
    bl = baseline_df[baseline_df["bl_dt"] <= cutoff_dt]
    if allowed_ids is not None:
        bl = bl[bl["subj_id"].isin(allowed_ids)]
    if len(bl) == 0:
        return pd.DataFrame(columns=list(baseline_df.columns) + ["time_m", "event"])

    valid_ids = set(bl["subj_id"])
    # Strictly restrict longitudinal records to visit_dt <= cutoff_dt
    lg = long_df[(long_df["visit_dt"] <= cutoff_dt) & (long_df["subj_id"].isin(valid_ids))]
    if len(lg) == 0:
        return pd.DataFrame(columns=list(baseline_df.columns) + ["time_m", "event"])

    # Join baseline date to compute positive follow-up months
    lg = lg.merge(bl[["subj_id", "bl_dt"]], on="subj_id", how="inner")
    lg["fu_m"] = (lg["visit_dt"] - lg["bl_dt"]).dt.days / 30.4375
    lg = lg[lg["fu_m"] > 0.0]
    if len(lg) == 0:
        return pd.DataFrame(columns=list(baseline_df.columns) + ["time_m", "event"])

    max_fu = lg.groupby("subj_id")["fu_m"].max()
    ev_sub = lg[lg["is_event"]]
    first_ev = ev_sub.groupby("subj_id")["fu_m"].min() if len(ev_sub) > 0 else pd.Series(dtype=float)

    surv = pd.DataFrame({"max_fu": max_fu}).join(first_ev.rename("ev_t"), how="left")
    surv["event"] = surv["ev_t"].notna().astype(np.int64)
    surv["time_m"] = np.where(surv["event"] == 1, surv["ev_t"], surv["max_fu"])
    surv = surv[["time_m", "event"]].reset_index()

    out = bl.merge(surv, on="subj_id", how="inner")
    out = out.sort_values("subj_id").reset_index(drop=True)
    return out


def build_adni_endpoints(adni_raw_dir: str, adni_extract_dir: str) -> Tuple[List[EndpointBundle], Dict]:
    """Build 4 enrolled ADNI survival endpoints (2005-2026) with 9 harmonized baseline covariates."""
    v26 = os.path.join(adni_extract_dir, "v2026_10", "csv")
    adsl = pd.read_csv(os.path.join(v26, "ADSL.csv"), low_memory=False)
    adsl["RID"] = pd.to_numeric(adsl["SUBJID"], errors="coerce")
    enrl = adsl[(adsl["ENRLFL"] == "Y") & (adsl["ORIGPROT"].isin(ADNI_PHASES))].copy()
    enrl = enrl.dropna(subset=["RID"]).drop_duplicates("RID")
    enrl["RID"] = enrl["RID"].astype(int)

    # Live DXSUM (matches DXSUM*.csv regardless of LONI download date)
    dx_candidates = sorted(glob.glob(os.path.join(adni_raw_dir, "DXSUM*.csv")))
    if not dx_candidates:
        raise FileNotFoundError(f"No DXSUM*.csv found under {adni_raw_dir}")
    dx = pd.read_csv(dx_candidates[-1], low_memory=False)
    dx["RID"] = pd.to_numeric(dx["RID"], errors="coerce")
    dx["visit_dt"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx["dx"] = pd.to_numeric(dx["DIAGNOSIS"], errors="coerce")
    dx = dx[dx["RID"].isin(enrl["RID"]) & dx["visit_dt"].notna() & dx["dx"].isin([1, 2, 3])].copy()
    dx["RID"] = dx["RID"].astype(int)
    dx = dx.sort_values(["RID", "visit_dt"])

    bl_dx = dx.groupby("RID", as_index=False).first().rename(columns={"visit_dt": "bl_dt", "dx": "bl_dx"})
    base = enrl[["RID", "USUBJID", "ORIGPROT", "AGE", "SEX", "EDUC", "APOE"]].merge(
        bl_dx[["RID", "bl_dt", "bl_dx"]], on="RID", how="inner"
    )

    # Backfill EDUC from PTDEMOG and APOE from APOERES if missing
    demog = pd.read_csv(os.path.join(v26, "PTDEMOG.csv"), usecols=["RID", "VISDATE", "PTEDUCAT", "PTGENDER"], low_memory=False)
    demog = demog.sort_values(["RID", "VISDATE"]).groupby("RID", as_index=False).first()
    apoe_tab = pd.read_csv(os.path.join(v26, "APOERES.csv"), usecols=["RID", "GENOTYPE"], low_memory=False).dropna(subset=["GENOTYPE"]).drop_duplicates("RID")
    base = base.merge(demog[["RID", "PTEDUCAT"]], on="RID", how="left").merge(apoe_tab, on="RID", how="left")

    base["age"] = _clean_range(base["AGE"], 40.0, 110.0)
    base["sex_male"] = np.where(base["SEX"] == "Male", 1.0, np.where(base["SEX"] == "Female", 0.0, np.nan))
    educ_a = _clean_range(base["EDUC"], 0.0, 30.0)
    educ_b = _clean_range(base["PTEDUCAT"], 0.0, 30.0)
    base["educ"] = educ_a.fillna(educ_b)
    apoe_a = _parse_apoe4_count(base["APOE"])
    apoe_b = _parse_apoe4_count(base["GENOTYPE"])
    base["apoe4_count"] = apoe_a.fillna(apoe_b)

    # Baseline CDR-SB, FAQ, TrailB from ADQS (ABLFL=='Y', fallback nearest <=180d)
    adqs = pd.read_csv(os.path.join(v26, "ADQS.csv"), usecols=["SUBJID", "PARAMCD", "AVAL", "ADT", "ABLFL"], low_memory=False)
    adqs["RID"] = pd.to_numeric(adqs["SUBJID"], errors="coerce")
    adqs = adqs[adqs["RID"].isin(base["RID"]) & adqs["PARAMCD"].isin(["CDRSB", "FAQTOTAL", "TRABSCOR"])].copy()
    adqs["RID"] = adqs["RID"].astype(int)
    adqs["visit_dt"] = pd.to_datetime(adqs["ADT"], errors="coerce")
    adqs["AVAL"] = pd.to_numeric(adqs["AVAL"], errors="coerce")
    adqs = adqs.dropna(subset=["AVAL", "visit_dt"])

    def _extract_adqs_bl(param: str, lo: float, hi: float) -> pd.Series:
        sub = adqs[adqs["PARAMCD"] == param].copy()
        sub["val"] = _clean_range(sub["AVAL"], lo, hi)
        sub = sub.dropna(subset=["val"]).merge(base[["bl_dt"]].reset_index(), on="RID", how="inner")
        sub["diff"] = (sub["visit_dt"] - sub["bl_dt"]).dt.days.abs()
        sub["is_abl"] = (sub["ABLFL"] == "Y").astype(int)
        # Prefer ABLFL=='Y', then smallest absolute date diff within 180d
        sub = sub[(sub["is_abl"] == 1) | (sub["diff"] <= 180)].sort_values(["RID", "is_abl", "diff"], ascending=[True, False, True])
        return sub.groupby("RID")["val"].first()

    base = base.set_index("RID")
    base["cdrsb"] = _extract_adqs_bl("CDRSB", 0.0, 18.0)
    base["faq"] = _extract_adqs_bl("FAQTOTAL", 0.0, 30.0)
    base["trailb"] = _extract_adqs_bl("TRABSCOR", 0.0, 300.0)

    # Baseline Trail A, Animals, and fallback Trail B from NEUROBAT (<=180d of bl_dt)
    nb = pd.read_csv(os.path.join(v26, "NEUROBAT.csv"), usecols=["RID", "VISDATE", "TRAASCOR", "TRABSCOR", "CATANIMSC"], low_memory=False)
    nb["RID"] = pd.to_numeric(nb["RID"], errors="coerce")
    nb = nb[nb["RID"].isin(base.index)].copy()
    nb["RID"] = nb["RID"].astype(int)
    nb["visit_dt"] = pd.to_datetime(nb["VISDATE"], errors="coerce")
    nb = nb.dropna(subset=["visit_dt"]).merge(base[["bl_dt"]].reset_index(), on="RID", how="inner")
    nb["diff"] = (nb["visit_dt"] - nb["bl_dt"]).dt.days.abs()
    nb = nb[nb["diff"] <= 180].sort_values(["RID", "diff"])

    for col_raw, col_out, lo, hi in [
        ("TRAASCOR", "traila", 0.0, 150.0),
        ("CATANIMSC", "animals", 0.0, 77.0),
        ("TRABSCOR", "trailb_nb", 0.0, 300.0),
    ]:
        s = _clean_range(nb[col_raw], lo, hi)
        tmp = nb.assign(_v=s).dropna(subset=["_v"]).groupby("RID")["_v"].first()
        if col_out == "trailb_nb":
            base["trailb"] = base["trailb"].fillna(tmp)
        else:
            base[col_out] = tmp

    # Also backfill baseline CDRSB/FAQ from raw CDR.csv / FAQ.csv if missing
    for fname, dcol, scol, col_out, lo, hi in [
        ("CDR.csv", "VISDATE", "CDRSB", "cdrsb", 0.0, 18.0),
        ("FAQ.csv", "VISDATE", "FAQTOTAL", "faq", 0.0, 30.0),
    ]:
        raw_t = pd.read_csv(os.path.join(v26, fname), usecols=["RID", dcol, scol], low_memory=False)
        raw_t["RID"] = pd.to_numeric(raw_t["RID"], errors="coerce")
        raw_t = raw_t[raw_t["RID"].isin(base.index)].copy()
        raw_t["RID"] = raw_t["RID"].astype(int)
        raw_t["visit_dt"] = pd.to_datetime(raw_t[dcol], errors="coerce")
        raw_t["_v"] = _clean_range(raw_t[scol], lo, hi)
        raw_t = raw_t.dropna(subset=["visit_dt", "_v"]).merge(base[["bl_dt"]].reset_index(), on="RID", how="inner")
        raw_t["diff"] = (raw_t["visit_dt"] - raw_t["bl_dt"]).dt.days.abs()
        raw_t = raw_t[raw_t["diff"] <= 180].sort_values(["RID", "diff"])
        base[col_out] = base[col_out].fillna(raw_t.groupby("RID")["_v"].first())

    base = base.reset_index()
    base["subj_id"] = base["RID"].map(lambda r: f"ADNI_{int(r):05d}")
    base["phase_ver"] = base["ORIGPROT"].astype(str)
    base["bl_year"] = base["bl_dt"].dt.year.astype(int)
    base["stratum"] = base["phase_ver"] + "_" + base["bl_dx"].astype(int).astype(str)
    base["is_eval"] = _deterministic_stratified_holdout(
        base["subj_id"], base["stratum"], seed_tag="LC01-adni-survival-eval-20261123", frac=0.30
    )

    rid_to_sid = dict(zip(base["RID"], base["subj_id"]))
    final_cut = pd.Timestamp("2026-11-20")

    # Longitudinal tables for CDRSB and FAQ (combine raw CDR/FAQ + ADQS for maximum coverage)
    cdr_raw = pd.read_csv(os.path.join(v26, "CDR.csv"), usecols=["RID", "VISDATE", "CDRSB"], low_memory=False)
    cdr_raw["RID"] = pd.to_numeric(cdr_raw["RID"], errors="coerce")
    cdr_raw = cdr_raw[cdr_raw["RID"].isin(rid_to_sid)].copy()
    cdr_raw["visit_dt"] = pd.to_datetime(cdr_raw["VISDATE"], errors="coerce")
    cdr_raw["val"] = _clean_range(cdr_raw["CDRSB"], 0.0, 18.0)
    cdr_adqs = adqs[adqs["PARAMCD"] == "CDRSB"][["RID", "visit_dt", "AVAL"]].rename(columns={"AVAL": "val"})
    cdr_long = pd.concat([cdr_raw[["RID", "visit_dt", "val"]], cdr_adqs], ignore_index=True).dropna()
    cdr_long["subj_id"] = cdr_long["RID"].astype(int).map(rid_to_sid)
    cdr_long = cdr_long.groupby(["subj_id", "visit_dt"], as_index=False)["val"].max()

    faq_raw = pd.read_csv(os.path.join(v26, "FAQ.csv"), usecols=["RID", "VISDATE", "FAQTOTAL"], low_memory=False)
    faq_raw["RID"] = pd.to_numeric(faq_raw["RID"], errors="coerce")
    faq_raw = faq_raw[faq_raw["RID"].isin(rid_to_sid)].copy()
    faq_raw["visit_dt"] = pd.to_datetime(faq_raw["VISDATE"], errors="coerce")
    faq_raw["val"] = _clean_range(faq_raw["FAQTOTAL"], 0.0, 30.0)
    faq_adqs = adqs[adqs["PARAMCD"] == "FAQTOTAL"][["RID", "visit_dt", "AVAL"]].rename(columns={"AVAL": "val"})
    faq_long = pd.concat([faq_raw[["RID", "visit_dt", "val"]], faq_adqs], ignore_index=True).dropna()
    faq_long["subj_id"] = faq_long["RID"].astype(int).map(rid_to_sid)
    faq_long = faq_long.groupby(["subj_id", "visit_dt"], as_index=False)["val"].max()

    dx_long = dx[["RID", "visit_dt", "dx"]].copy()
    dx_long["subj_id"] = dx_long["RID"].map(rid_to_sid)

    keep_cols = ["subj_id", "bl_dt", "stratum", "phase_ver", "bl_year", "is_eval"] + FEATURES_9

    bundles = [
        EndpointBundle(
            cohort="ADNI",
            endpoint_id="ADNI_S1_mci_to_dem",
            label="ADNI: MCI to Dementia",
            baseline_df=base[base["bl_dx"] == 2][keep_cols].copy().reset_index(drop=True),
            long_df=dx_long[["subj_id", "visit_dt"]].assign(is_event=(dx_long["dx"] == 3)),
            final_cutoff=final_cut,
        ),
        EndpointBundle(
            cohort="ADNI",
            endpoint_id="ADNI_S2_cn_to_imp",
            label="ADNI: CN to MCI/Dementia",
            baseline_df=base[base["bl_dx"] == 1][keep_cols].copy().reset_index(drop=True),
            long_df=dx_long[["subj_id", "visit_dt"]].assign(is_event=dx_long["dx"].isin([2, 3])),
            final_cutoff=final_cut,
        ),
        EndpointBundle(
            cohort="ADNI",
            endpoint_id="ADNI_S3_cdrsb45",
            label="ADNI: CDR-SB >= 4.5",
            baseline_df=base[base["cdrsb"].notna() & (base["cdrsb"] < 4.5)][keep_cols].copy().reset_index(drop=True),
            long_df=cdr_long[["subj_id", "visit_dt"]].assign(is_event=(cdr_long["val"] >= 4.5)),
            final_cutoff=final_cut,
        ),
        EndpointBundle(
            cohort="ADNI",
            endpoint_id="ADNI_S4_faq9",
            label="ADNI: FAQ >= 9",
            baseline_df=base[base["faq"].notna() & (base["faq"] < 9.0)][keep_cols].copy().reset_index(drop=True),
            long_df=faq_long[["subj_id", "visit_dt"]].assign(is_event=(faq_long["val"] >= 9.0)),
            final_cutoff=final_cut,
        ),
    ]

    meta = {
        "cohort": "ADNI",
        "n_enrolled": int(len(base)),
        "n_eval_pool": int(base["is_eval"].sum()),
        "n_dev_pool": int((~base["is_eval"]).sum()),
        "feature_missing_pct": {f: round(float(base[f].isna().mean() * 100.0), 2) for f in FEATURES_9},
        "by_phase": base["phase_ver"].value_counts().to_dict(),
    }
    return bundles, meta


def build_nacc_endpoints(nacc_parquet: str) -> Tuple[List[EndpointBundle], Dict]:
    """Build 4 NACC Freeze 75 survival endpoints (2005-2026) with 9 harmonized baseline covariates."""
    cols = [
        "NACCID", "NACCVNUM", "VISITDATE", "FORMVER", "NACCAGE", "NACCSEX", "EDUC",
        "NACCNE4S", "NACCUDSD", "CDRSUM", "TRAILA", "TRAILB", "ANIMALS",
    ] + FAQ_ITEMS_NACC
    df = pd.read_parquet(nacc_parquet, columns=cols)
    df["visit_dt"] = pd.to_datetime(df["VISITDATE"], errors="coerce")
    df = df.dropna(subset=["NACCID", "visit_dt"]).sort_values(["NACCID", "NACCVNUM"])

    # Clean clinical measurements
    df["cdrsb"] = _clean_range(df["CDRSUM"], 0.0, 18.0)
    faq_mat = np.column_stack([_clean_range(df[c], 0.0, 3.0).to_numpy(dtype=np.float64) for c in FAQ_ITEMS_NACC])
    faq_valid_cnt = np.sum(np.isfinite(faq_mat), axis=1)
    with np.errstate(invalid="ignore"):
        faq_sum = np.nansum(faq_mat, axis=1)
        faq_scaled = np.where(faq_valid_cnt >= 8, (faq_sum / np.maximum(faq_valid_cnt, 1)) * 10.0, np.nan)
    df["faq"] = faq_scaled

    # Baseline visit = first visit per NACCID (NACCVNUM == 1)
    bl = df.groupby("NACCID", as_index=False).first().copy()
    bl["subj_id"] = bl["NACCID"].astype(str)
    bl["bl_dt"] = bl["visit_dt"]
    bl["bl_year"] = bl["bl_dt"].dt.year.astype(int)
    bl["phase_ver"] = "UDS" + bl["FORMVER"].fillna(0).astype(int).astype(str)
    bl["bl_dx"] = pd.to_numeric(bl["NACCUDSD"], errors="coerce").fillna(-1).astype(int)

    bl["age"] = _clean_range(bl["NACCAGE"], 18.0, 120.0)
    sex_num = pd.to_numeric(bl["NACCSEX"], errors="coerce")
    bl["sex_male"] = np.where(sex_num == 1, 1.0, np.where(sex_num == 2, 0.0, np.nan))
    bl["educ"] = _clean_range(bl["EDUC"], 0.0, 36.0)
    bl["apoe4_count"] = _clean_range(bl["NACCNE4S"], 0.0, 2.0)
    bl["traila"] = _clean_range(bl["TRAILA"], 0.0, 150.0)
    bl["trailb"] = _clean_range(bl["TRAILB"], 0.0, 300.0)
    bl["animals"] = _clean_range(bl["ANIMALS"], 0.0, 77.0)

    bl["stratum"] = bl["phase_ver"] + "_" + bl["bl_dx"].astype(str)
    bl["is_eval"] = _deterministic_stratified_holdout(
        bl["subj_id"], bl["stratum"], seed_tag="LC01-nacc-survival-eval-20261123", frac=0.30
    )

    final_cut = pd.Timestamp("2026-08-20")
    keep_cols = ["subj_id", "bl_dt", "stratum", "phase_ver", "bl_year", "is_eval"] + FEATURES_9

    lg_base = pd.DataFrame({
        "subj_id": df["NACCID"].astype(str),
        "visit_dt": df["visit_dt"],
        "dx": pd.to_numeric(df["NACCUDSD"], errors="coerce"),
        "cdrsb": df["cdrsb"],
        "faq": df["faq"],
    })

    bundles = [
        EndpointBundle(
            cohort="NACC",
            endpoint_id="NACC_S1_mci_to_dem",
            label="NACC: MCI to Dementia",
            baseline_df=bl[bl["bl_dx"] == 3][keep_cols].copy().reset_index(drop=True),
            long_df=lg_base.loc[lg_base["dx"].notna(), ["subj_id", "visit_dt"]].assign(
                is_event=(lg_base.loc[lg_base["dx"].notna(), "dx"] == 4)
            ),
            final_cutoff=final_cut,
        ),
        EndpointBundle(
            cohort="NACC",
            endpoint_id="NACC_S2_cn_to_imp",
            label="NACC: Normal to MCI/Dementia",
            baseline_df=bl[bl["bl_dx"] == 1][keep_cols].copy().reset_index(drop=True),
            long_df=lg_base.loc[lg_base["dx"].notna(), ["subj_id", "visit_dt"]].assign(
                is_event=lg_base.loc[lg_base["dx"].notna(), "dx"].isin([3, 4])
            ),
            final_cutoff=final_cut,
        ),
        EndpointBundle(
            cohort="NACC",
            endpoint_id="NACC_S3_cdrsb45",
            label="NACC: CDR-SB >= 4.5",
            baseline_df=bl[bl["cdrsb"].notna() & (bl["cdrsb"] < 4.5)][keep_cols].copy().reset_index(drop=True),
            long_df=lg_base.loc[lg_base["cdrsb"].notna(), ["subj_id", "visit_dt"]].assign(
                is_event=(lg_base.loc[lg_base["cdrsb"].notna(), "cdrsb"] >= 4.5)
            ),
            final_cutoff=final_cut,
        ),
        EndpointBundle(
            cohort="NACC",
            endpoint_id="NACC_S4_faq9",
            label="NACC: FAQ >= 9",
            baseline_df=bl[bl["faq"].notna() & (bl["faq"] < 9.0)][keep_cols].copy().reset_index(drop=True),
            long_df=lg_base.loc[lg_base["faq"].notna(), ["subj_id", "visit_dt"]].assign(
                is_event=(lg_base.loc[lg_base["faq"].notna(), "faq"] >= 9.0)
            ),
            final_cutoff=final_cut,
        ),
    ]

    meta = {
        "cohort": "NACC",
        "n_enrolled": int(len(bl)),
        "n_eval_pool": int(bl["is_eval"].sum()),
        "n_dev_pool": int((~bl["is_eval"]).sum()),
        "feature_missing_pct": {f: round(float(bl[f].isna().mean() * 100.0), 2) for f in FEATURES_9},
        "by_phase": bl["phase_ver"].value_counts().to_dict(),
    }
    return bundles, meta


def load_adnimerge_snapshot_df(csv_path: str) -> pd.DataFrame:
    """Normalize ADNIMERGE 2023-09 or 2025-07 table into standardized columns for real-snapshot fidelity audit."""
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.rename(columns={
        "DX.bl": "DX_bl",
        "CDRSB.bl": "CDRSB_bl",
        "FAQ.bl": "FAQ_bl",
        "MMSE.bl": "MMSE_bl",
        "LDELTOTAL.bl": "LDELTOTAL_BL",
    })
    df["RID"] = pd.to_numeric(df["RID"], errors="coerce")
    df["visit_dt"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df = df.dropna(subset=["RID", "visit_dt"]).copy()
    df["RID"] = df["RID"].astype(int)
    df["subj_id"] = df["RID"].map(lambda r: f"ADNI_{r:05d}")
    # Map DX string to 1=CN, 2=MCI, 3=Dementia
    dx_s = df["DX"].astype(str).str.strip()
    df["dx_num"] = dx_s.map({"CN": 1, "MCI": 2, "Dementia": 3})
    df["cdrsb"] = _clean_range(df["CDRSB"], 0.0, 18.0)
    df["faq"] = _clean_range(df["FAQ"], 0.0, 30.0)
    return df
