# Decisions Log - Vancouver Building Permits Project

## 1. Project scope

| # | Decision | Reason |
|---|---|---|
| 1 | Use the City of Vancouver **Issued Building Permits** dataset (52,199 rows, 20 columns, opendata.vancouver.ca). | Quick baseline models were stronger and more stable on this data than on the retractions alternative, with lower leakage risk and a balanced target. |
| 2 | **Classification only**, with 4-5 algorithms (approved by the professor). | Meets the "at least two models, one predictive" requirement through the approved substitution. |
| 3 | **70/30 stratified train/test split**, `random_state=42`, made once and shared by all models. | Professor's requirement. Stratifying keeps the on-time/delayed balance equal in both sets; one shared split makes the model comparison fair. |
| 4 | Tuning and cross-validation use the **training set only**; the test set is used once for final scores. | Prevents leakage and over-optimistic results. |

## 2. Target definition

| # | Decision | Reason |
|---|---|---|
| 5 | Target `on_time` = 1 if `PermitElapsedDays` <= **85 days** (the median of the full file), else 0. | Median split gives balanced classes and an easy-to-explain rule. Named "on_time" so the target reads as a business outcome. |
| 6 | The 85-day threshold was fixed on the full file **before** the 2017-2025 filter. | After filtering, the median of `elapsed_days` is 83, but the threshold stays at 85 so the target definition does not change with the filter. Resulting split: 51.2% on time / 48.8% delayed. |
| 7 | Positive class = on time. Report metrics for the delayed class (0) as well, or use macro-averaged F1. | The project question is about spotting delays; class balance makes either choice valid, but the choice must be stated. |

## 3. Cleaning

| # | Decision | Reason |
|---|---|---|
| 8 | Keep only permits **created 2017-2025** (removed 4,761 rows; 47,438 remain, about 91%). | The file contains only permits *issued* from 2017 onward and was extracted in Sept 2026. Pre-2017 creations appear only if they were very slow (left truncation: 0-14% on time); 2026 creations appear only if already issued (right censoring: 79.9% on time). Keeping them would let the model learn the data window instead of permit behaviour. |
| 9 | Drop `IssueDate`, `IssueYear`, `YearMonth` and `PermitNumber` from the modelling data. | `IssueDate` minus the created date reproduces `PermitElapsedDays` exactly (verified: match = 1.0), so it leaks the target. The permit-number prefix hints at the type of work. |
| 10 | Keep `elapsed_days` in the clean file for **EDA only**; never use it as a model feature. | It defines the target. |
| 11 | Drop identifier and free-text columns: Address, Applicant, ApplicantAddress, BuildingContractor, BuildingContractorAddress, ProjectDescription, SpecificUseCategory, Geom, geo_point_2d. | High-cardinality identifiers or duplicated information; not useful as classification features. Coordinates were extracted from `geo_point_2d` first. |
| 12 | `ProjectValue`: add `value_is_zero` flag and `log_value = log1p(value)`; keep all rows. | Value is extremely skewed (median $50,000, max $1.47 billion). Zeros are frequent (15.6% of the 2017-2025 data) and behave differently (67.3% on time vs 48.2%), so they are flagged, not dropped. |
| 13 | Fill missing `PermitCategory`, `GeoLocalArea` and `PropertyUse` with "Unknown". | Missingness is large for PermitCategory (42.6%) and may be informative; dropping rows would lose almost half the data. |
| 14 | Replace `BuildingContractor` with a `has_contractor` flag. | 38% missing and a high-cardinality name; whether a contractor is listed carries signal (43.4% on time with a contractor vs 64.2% without). |
| 15 | Group `PropertyUse` into the top 20 values plus "Other". | 154 distinct values; the top 20 cover 98.6% of rows. |
| 16 | Extract `created_year`, `created_month`, `lat`, `lon`; leave 348 missing coordinates as NaN. | Application timing is known at application time. Coordinates are mainly for maps and clustering; `area` already captures location for the classifiers. |
| 17 | No duplicates, negative values or unparseable dates were found. Zero-day permits (225) are kept. | Zero-day permits are plausible (same-day issue). |

## 4. Open decisions (to settle during modelling)

| # | Question | Plan |
|---|---|---|
| A | Include `created_year` as a feature? | Run the models with and without it and report both; the on-time rate drifts by year (38% in 2021, 59% in 2025). |
| B | Group the very small `type_of_work` classes (Outdoor Uses has 59 rows)? | Decide after EDA. |
| C | Use `lat`/`lon` in the classifiers? | Probably not; keep for clustering and maps. |
| D | Add clustering as a bonus section? | Decide once the five classifiers are done. |

## 5. Known limitations (for the presentation)

- Permits created late in 2025 may still be partly censored (2025 on-time rate: 59%).
- `type_of_work` is a very strong predictor (on-time rate from 8% for New Building to 92% for Temporary Building), so feature importance will lean on it.
- Predictions describe statistical association in issued permits, not causes of delay.
- Only issued permits are included; applications that were withdrawn or are still pending are not in the data.
