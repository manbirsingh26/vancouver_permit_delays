# Decisions Log - Vancouver Building Permits Project
Last updated: 2026-11-08

## 1. Project scope

| # | Decision | Reason |
|---|---|---|
| 1 | Use the City of Vancouver **Issued Building Permits** dataset (52,199 rows, 20 columns, opendata.vancouver.ca). | Quick baseline models were stronger and more stable on this data than on the retractions alternative, with lower leakage risk and a balanced target. |
| 2 | **Classification as the main analysis** (4 algorithms, approved by the professor), plus **K-Means clustering** as a second, descriptive analysis. | Meets the "at least two models, one predictive" requirement; clustering adds a different view (which neighbourhoods behave alike). |
| 3 | **70/30 stratified train/test split**, `random_state=42`, made once and shared by all models. | Professor's requirement. Stratifying keeps the on-time/delayed balance equal in both sets; one shared split makes the model comparison fair. |
| 4 | Tuning and cross-validation use the **training set only**; the test set is used once per model for final scores. | Prevents leakage and over-optimistic results. |

## 2. Target definition

| # | Decision | Reason |
|---|---|---|
| 5 | Target `on_time` = 1 if `PermitElapsedDays` <= **85 days** (the median of the full file), else 0. | Median split gives balanced classes and an easy-to-explain rule. Named "on_time" so the target reads as a business outcome. |
| 6 | The 85-day threshold was fixed on the full file **before** the 2017-2025 filter. | After filtering, the median of `elapsed_days` is 83, but the threshold stays at 85 so the target definition does not change with the filter. Resulting split: 51.2% on time / 48.8% delayed. |
| 7 | Positive class = on time. Metrics are reported for **both classes**, plus macro-averaged F1. | The project question is about spotting delays, so the delayed class matters as much as the on-time class. |

## 3. Cleaning

| # | Decision | Reason |
|---|---|---|
| 8 | Keep only permits **created 2017-2025** (removed 4,761 rows; 47,438 remain, about 91%). | The file contains only permits *issued* from 2017 onward and was extracted in Sept 2026. Pre-2017 creations appear only if they were very slow (left truncation: 0-14% on time); 2026 creations appear only if already issued (right censoring: 79.9% on time). Keeping them would let the model learn the data window instead of permit behaviour. |
| 9 | Drop `IssueDate`, `IssueYear`, `YearMonth` and `PermitNumber` from the modelling data. | `IssueDate` minus the created date reproduces `PermitElapsedDays` exactly (verified: match = 1.0), so it leaks the target. The permit-number prefix hints at the type of work. |
| 10 | Keep `elapsed_days` in the clean file for **EDA and clustering only**; never use it as a classification feature. | It defines the target. |
| 11 | Drop identifier and free-text columns: Address, Applicant, ApplicantAddress, BuildingContractor, BuildingContractorAddress, ProjectDescription, SpecificUseCategory, Geom, geo_point_2d. | High-cardinality identifiers or duplicated information; not useful as classification features. Coordinates were extracted from `geo_point_2d` first. |
| 12 | `ProjectValue`: add `value_is_zero` flag and `log_value = log1p(value)`; keep all rows. | Value is extremely skewed (median $50,000, max $1.47 billion). Zeros are frequent (15.6% of the 2017-2025 data) and behave differently (67.3% on time vs 48.2%), so they are flagged, not dropped. |
| 13 | Fill missing `PermitCategory`, `GeoLocalArea` and `PropertyUse` with "Unknown". | Missingness is large for PermitCategory (42.6%) and may be informative; dropping rows would lose almost half the data. |
| 14 | Replace `BuildingContractor` with a `has_contractor` flag. | 38% missing and a high-cardinality name; whether a contractor is listed carries signal (43.4% on time with a contractor vs 64.2% without). |
| 15 | Group `PropertyUse` into the top 20 values plus "Other". | 154 distinct values; the top 20 cover 98.6% of rows. |
| 16 | Extract `created_year`, `created_month`, `lat`, `lon`; leave 348 missing coordinates as NaN. | Application timing is known at application time. Coordinates are used only for maps and clustering; `area` already captures location for the classifiers. |
| 17 | No duplicates, negative values or unparseable dates were found. Zero-day permits (225) are kept. | Zero-day permits are plausible (same-day issue). |

## 4. Modelling setup

| # | Decision | Reason |
|---|---|---|
| 18 | **Features (9):** 4 categorical (`type_of_work`, `property_use`, `area`, `permit_category`) + 5 numeric (`log_value`, `value_is_zero`, `has_contractor`, `created_month`, `created_year`). | Only information known at application time. `created_year` is controlled by the `USE_YEAR` switch. |
| 19 | **Preprocessing:** one-hot encoding for categorical features, standardisation for numeric ones, fitted inside pipelines on the training data only. Gradient boosting uses ordinal encoding with native categorical support instead. | Keeps the test set unseen during fitting, including inside cross-validation. |
| 20 | **Evaluation:** 5-fold stratified CV on the training set for stability; final scores once on the test set. Metrics: accuracy, macro F1, ROC-AUC, and precision/recall/F1 for both classes, with a confusion matrix and ROC curve per model. | Classification metrics suit a binary target (R² does not apply). Both classes are reported because spotting delays is the goal. |
| 21 | Models live in `03_models.ipynb` and load the saved split from `data/split/`; the split is made only in `02_main_analysis.ipynb`. | Keeps one shared test set for all models and avoids re-splitting by accident. |
| 22 | Compare all models with **two simple baselines**: always predicting the majority class (accuracy 0.512) and predicting from type of work alone (0.801). | Shows how much the models add beyond the single strongest feature. |

## 5. Classification models

| # | Decision | Reason |
|---|---|---|
| 23 | **Logistic regression** as the baseline model (`max_iter=1000`, no tuning, no class weighting). | Simple and interpretable; classes are balanced, so no weighting is needed. Test accuracy 0.838, ROC-AUC 0.892. |
| 24 | **Decision tree** tuned with GridSearchCV (max_depth 4-15, min_samples_leaf 1-50, gini/entropy; 48 combinations, ROC-AUC). Best: depth 15, min_samples_leaf 50, gini. | An unrestricted tree overfits (train 0.977, test 0.785, depth 47). A depth curve up to 20 was also checked. Test accuracy 0.842, ROC-AUC 0.905. |
| 25 | **Random forest** tuned with RandomizedSearchCV, n_iter = 14 of 100 combinations (n_estimators 200/300, max_depth 10/15/20/25/None, min_samples_leaf 1/5/10/20/30, max_features sqrt/0.3). Best: 300 trees, max_depth 25, min_samples_leaf 5, max_features 0.3. | A full grid would take close to an hour; a randomized search is standard for slow models. Test accuracy 0.851, ROC-AUC 0.914. |
| 26 | **Gradient boosting** (`HistGradientBoostingClassifier`) tuned with RandomizedSearchCV, n_iter = 15. Best: learning_rate 0.03, max_iter 400, max_leaf_nodes 63, min_samples_leaf 20, l2_regularization 0.0. | Chosen as the fourth algorithm over Naive Bayes (assumes independent features, but type of work and permit category are strongly related) and kNN (distance across about 60 one-hot columns is not meaningful, and it is slow on 33k rows). Built into scikit-learn, so no extra install. Test accuracy 0.848, ROC-AUC 0.912. |
| 27 | **Feature importance** checked two ways: impurity-based (decision tree, random forest) and permutation importance on the test set (random forest, gradient boosting; ROC-AUC drop, 10 repeats). | Impurity-based importance favours continuous and many-category features; permutation importance on unseen data is more reliable. Both methods agree: type of work and permit category lead, then project value and application year. |
| 28 | **Conclusion:** random forest performed best; gradient boosting was essentially tied. | The gap (0.003 accuracy, 0.002 AUC) is about the size of the CV standard deviation, so the two should be presented as close. |

## 6. Clustering

| # | Decision | Reason |
|---|---|---|
| 29 | Use **clustering** rather than association rules. | The data is one row per permit, not basket-shaped; association rules would mostly restate the obvious patterns already shown by EDA and the classifiers. |
| 30 | Cluster the **22 neighbourhoods** (aggregated profiles), not individual permits. "Unknown" area (379 permits) excluded. | Clustering individual permits would mostly reproduce the type-of-work groups. Neighbourhood profiles answer a new question: which areas behave alike. |
| 31 | Clustering features: **log permits, share of alterations, log median project value, on-time rate**, standardised. | The excluded measures (share of new buildings, share of demolitions, median days) correlate at 0.95 or more with the included ones and would count the same idea several times. Using processing time here is fine because clustering is not a classifier. |
| 32 | **K-Means with k = 3** (`n_init=20`, `random_state=42`), with a Ward-linkage dendrogram, elbow chart and silhouette score used to choose k. | k = 2 has the best silhouette (0.477) but only separates central from residential areas. The dendrogram's next split gives k = 3 (silhouette 0.305). k = 5 scores almost the same (0.308) but splits off single neighbourhoods (e.g. Downtown alone). |
| 33 | Clusters: **central/commercial** (6 areas, 69% on time), **quieter residential** (9 areas, 41%), **busy residential** (7 areas, 37%). | Shown in a profile table, a PCA plot (2 components keep 83% of the variation) and a map. Silhouette 0.305 means the two residential groups overlap somewhat; presented as a description, not sharply separated groups. |

## 7. Open decisions

| # | Question | Status |
|---|---|---|
| A | Include `created_year` as a feature? | Included in all models. It ranks 4th in permutation importance. Optional re-run with `USE_YEAR = False` still to decide. |
| B | Group the very small `type_of_work` classes (Outdoor Uses has 59 rows)? | Not grouped; `handle_unknown="ignore"` covers any category missing from a split. |
| C | Use `lat`/`lon` in the classifiers? | Not used; kept for maps and clustering only. |
| D | Add clustering as a second analysis? | Done (section 6). |

## 8. Known limitations (for the presentation)

- Permits created late in 2025 may still be partly censored (2025 on-time rate: 59%).
- `type_of_work` is a very strong predictor (on-time rate from 8% for New Building to 92% for Temporary Building); type of work alone already reaches 80% accuracy.
- The models rely partly on `created_year`, which may reflect changing workloads at the city and may not hold for future years.
- Predictions and importances describe statistical associations in issued permits, not causes of delay.
- Only issued permits are included; applications that were withdrawn or are still pending are not in the data.
- Clustering uses only 22 neighbourhoods and has a moderate silhouette score (0.305).