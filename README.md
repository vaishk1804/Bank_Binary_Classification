### Bank Marketing — Binary Classification *(LogReg · Random Forest · XGBoost)*

Predict whether a customer subscribes to a term deposit (target `y ∈ {0,1}`) using the Kaggle Bank Marketing dataset. This project covers end-to-end preprocessing and a three-model comparison focused on robust, leakage-aware evaluation.

---

### Project Snapshot

- **Train:** 750,000 rows × 18 columns (`train.csv`)
- **Test:** 250,000 rows × 17 columns (`test.csv`)
- **Target:** `y` (`0 = no`, `1 = yes`)
- **Engineered table:** `Bank_cleaned.csv` *(produced after preprocessing)*

---

### Problem & Goal

- **Objective:** Classify likelihood of term deposit subscription to support targeted outreach.  
- **Challenges:** Class imbalance and potential target leakage via `duration`.  
- **Success Criteria:** Strong ranking metrics (**ROC AUC** / **PR AUC**) with practical **F1**/**Recall** trade-offs for campaign use.

---

### Data & Columns *(Raw)*

`id, age, job, marital, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome, y`

**Notes**
- No missing values by column; many **"unknown"** in `poutcome`.
- `duration` is known only **after** the call (leakage) and is therefore **excluded** from modeling.

---

### Preprocessing

- **Dropped columns (train & test):** `poutcome`, `id`, `duration` *(to avoid leakage)*.
- **Outlier review (IQR):** scanned `pdays`, `previous`, `balance`, `campaign`, `age` for reporting; no row removals.
- **Feature transforms (skew reduction):**
  - `balance_log` from `balance` *(lower-bounded at `0`)*
  - `campaign_log` from `campaign`
  - `previous_log` from `previous`
- **Sentinel feature:** `pdays_missing` indicating `pdays = -1`.
- **Dropped originals after engineering:** `previous`, `pdays`, `campaign`, `balance`.
- **Binary mappings:** `default`, `housing`, `loan` mapped from *“yes/no”* to `1/0`.
- **Cyclical time encoding:** created `day_sin` and `day_cos` from `day + month`; dropped intermediate time columns.
- **Categorical encoding:** One-Hot for `job`, `marital`, `education`, `contact`.
- **Output:** `Bank_cleaned.csv` containing engineered features and `y`.  
  *Ignore the CSV index column if saved as `Unnamed: 0`.*

---

### Modeling Approach

- **Models compared:** Logistic Regression *(balanced class weights)*, Random Forest *(balanced class weights)*, XGBoost *(with positive-class weighting from class ratio)*.
- **Split:** 80/20 train–test with shuffle and fixed random state; **stratification** to preserve label balance.
- **Evaluation:** **Accuracy**, **Precision**, **Recall**, **F1**, **ROC AUC**, and **ROC curves** for visual comparison.  
  *Accuracy can be misleading under imbalance; emphasize **PR/ROC** curves and **F1/Recall** when selecting an operating point.*

