# Data Understanding & Preparation for Airbnb Price Prediction

A data understanding and preparation pipeline for an NYC Airbnb pricing model — covering label engineering, outlier handling, missing-value imputation, one-hot encoding, and correlation-based feature exploration. Built as part of a Break Through Tech AI Studio "ML Life Cycle: Data Understanding and Data Preparation" lab.

## Table of Contents
- [Business Brief](#business-brief)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Findings](#findings)
- [Analysis](#analysis)
- [AI Usage Reflection](#ai-usage-reflection)
- [How to Run](#how-to-run)
- [Repository Contents](#repository-contents)

## Business Brief

**Company & Context:** NestIQ is a data and analytics company that builds machine learning models to help real estate and short-term rental clients make better pricing decisions.

**Business Challenge:** NestIQ's client, an NYC short-term rental operator, currently sets prices manually or with simple rules — a base price adjusted for bedrooms, room type, host rating, and location. This works as a starting point but doesn't scale as the business grows.

**Business Goal:** Build a pricing model that learns the relationship between listing features and price from the client's current listings data, and accurately predicts the price of a new listing.

**Role & Task:** As a junior ML engineer on NestIQ's Pricing Model team, the assignment was to explore an Airbnb NYC listings dataset and prepare it for modeling — a regression problem — making and documenting decisions about missing values, outliers, and feature engineering along the way.

## Project Objectives

- Load the Airbnb listings dataset, inspect its structure, and define the regression label (`price`).
- Convert the label from a currency-formatted string to a numeric type suitable for modeling.
- Identify and drop features unsuitable for modeling (e.g., URL columns).
- Handle outliers in the label via winsorization.
- Identify and impute missing values in numeric features, while preserving missingness as its own signal.
- One-hot encode categorical text features.
- Explore correlations between features and the label, and visualize the strongest relationships.

## Dataset

- **Source:** `airbnbData.csv` — raw NYC Airbnb "listings" data.
- **Label:** `price` (originally a string like `$120` or `$2,500`), cleaned into a numeric column and then winsorized into `label_price`.
- **Features:** All non-URL columns in the dataset, including numeric fields (`accommodates`, `bedrooms`, `bathrooms`, `beds`, `host_listings_count`, `host_total_listings_count`, etc.) and categorical fields (`room_type`, `host_response_time`, `host_identity_verified`, and others).

## Methodology

1. **Label definition & cleaning:** Inspected the `price` column, found it stored as strings (e.g., `$120`), and converted it to `float` by stripping `$` and `,` characters.
2. **Feature triage:** Identified and dropped all columns containing `url` in their name, since URLs carry no predictive signal for price.
3. **Outlier handling:** Created a new label column, `label_price`, by winsorizing the top and bottom 1% of `price` values using `scipy.stats.mstats.winsorize()`, and verified the transformation changed the underlying values.
4. **Missing value imputation:**
   - Identified all columns with missing values (`nan_detected`) and narrowed the list to numeric columns only (`is_int_or_float`), combining both into `to_impute`.
   - Selected five columns for imputation: `host_listings_count`, `host_total_listings_count`, `bathrooms`, `bedrooms`, `beds`.
   - Preserved missingness as a signal by creating `<column>_na` boolean flag columns before imputing.
   - Filled missing values in the selected columns with each column's mean. (Note: `bathrooms` retained missing values post-imputation because the column contained no non-null values to average in the first place.)
5. **One-hot encoding:**
   - Filled `NaN` values in `host_response_time` with the string `'unavailable'`, then one-hot encoded it with `pd.get_dummies()` and merged the result back into `df`.
   - Repeated the same fill-then-encode pattern for `room_type`.
   - Surveyed remaining `object`-typed columns and their unique-value counts to identify further one-hot encoding candidates (e.g., `host_listings_count_na`, `host_total_listings_count_na`, `bathrooms_na`, `bedrooms_na`, `beds_na`, `host_identity_verified`) based on having a small number of unique values.
6. **Correlation exploration:**
   - Computed a full correlation matrix (`df.corr()`) and extracted the column for `label_price`.
   - Sorted correlations in descending order and identified the top two feature correlates (excluding `label_price` itself and the pre-winsorized `price` column).
   - Built a `seaborn` KDE pairplot of `label_price` against its two strongest correlates (`accommodates` and `bedrooms`).

## Findings

- **Label transformation:** `price` was successfully converted from a currency string to a numeric type, and winsorization measurably changed the top/bottom 1% of values (confirmed via a nonzero difference between `price` and `label_price`).
- **Missing data:** Several numeric columns (`host_listings_count`, `host_total_listings_count`, `bathrooms`, `bedrooms`, `beds`) had missing values, imputed with column means and flagged with `_na` indicator columns to preserve information about what was originally missing.
- **Top correlates:** `accommodates` and `bedrooms` were the two features most correlated with `label_price`, though the relationship was weak overall — visualized via a KDE pairplot.

## Analysis

**Risks of imputation and winsorization:** Replacing missing values with column means can dilute the true variability in the data and mask relationships that would otherwise inform the model, potentially skewing NestIQ's price estimates toward the average rather than reflecting a listing's actual characteristics. Winsorizing extreme prices reduces the influence of outliers but also risks flattening genuinely high- or low-value listings that the model should learn to price correctly.

**Feature correlation & real-world gaps:** `accommodates` and `bedrooms` showed only a weak correlation with `label_price`, suggesting neither feature alone is a strong standalone predictor and that price is likely driven by a more complex combination of factors. One real-world factor missing from this dataset entirely is rental period/length of stay — longer stays could command different pricing than short ones, and without that signal, NestIQ's model may miss a meaningful driver of price.

**Unstructured text & remaining prep:** Columns like `name` and `description` contain unstructured text that could hold valuable pricing signal (e.g., property type or amenities mentioned in the listing description) but would need NLP techniques to be usable. Beyond that, remaining categorical features still need one-hot encoding, and a rental-period feature (if available) would need to be engineered before the dataset is fully ready for modeling.

## How to Run

1. Ensure the following are installed: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`.
2. Place `airbnbData.csv` in a `data/` subdirectory relative to the notebook.
3. Open `DataUnderstandingAndPreparation.ipynb` in Jupyter and run all cells in order.
4. Review the printed correlation values and the KDE pairplot in the final cells.

```bash
pip install pandas numpy scipy matplotlib seaborn
jupyter notebook DataUnderstandingAndPreparation.ipynb
```

## Repository Contents

- `DataUnderstandingAndPreparation.ipynb` — main notebook with data cleaning, imputation, encoding, correlation analysis, and written reflections
- `data/airbnbData.csv` — raw benchmark dataset (not included; place locally)
- `README.md` — this file
