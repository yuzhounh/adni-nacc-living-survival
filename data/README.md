# Data Directory

This repository contains **no raw ADNI or NACC data**, participant-level derived tables, fitted model checkpoints, or individual-level predictions.

## Required External Inputs (Under Approved Data Use Agreements)

1. **ADNI (Alzheimer's Disease Neuroimaging Initiative)** via the [LONI Image and Data Archive](https://ida.loni.usc.edu):
   - Archived `ADNIMERGE` (2023-09 release CSV) and `DXSUM` live diagnostic summary CSV placed in `data/raw/adni_route_a/` (or configured via `ADNI_RAW_DIR` / `--adni-raw-dir`).
   - Extracted CSV tables from `ADNIMERGE` (July 2025 package) and `ADNIMERGE2` (October 2026 package: `ADSL`, `ADQS`, `PTDEMOG`, `APOERES`, `CDR`, `FAQ`, `NEUROBAT`, `UCSFFSX`, `UCSFFSX51`, `UCSFFSX6`, `UCSFFSX7`, `UCBERKELEY_AMY_6MM`, `UCBERKELEY_TAU_6MM`, `UPENNBIOMK_ROCHE_ELECSYS`, `UPENN_PLASMA_FUJIREBIO_QUANTERIX`) placed under `data/raw/adni_extract/v2025_07/csv/` and `data/raw/adni_extract/v2026_10/csv/` (or configured via `ADNI_EXTRACT_DIR` / `--adni-extract-dir`).

2. **NACC Uniform Data Set (Freeze 75)** via the [National Alzheimer's Coordinating Center](https://naccdata.org):
   - Investigator longitudinal file converted to Parquet at `data/raw/nacc/investigator_nacc75.parquet` (or configured via `NACC_PARQUET` / `--nacc-parquet`).

All files under `data/raw/`, `data/derived/`, and `outputs/` are excluded by `.gitignore`.
