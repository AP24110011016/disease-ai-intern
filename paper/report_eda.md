# Exploratory Data Analysis (EDA) Report

## Data Source

Dataset: Pima Indians Diabetes Dataset

The dataset contains medical information about female patients and is used to predict whether a patient has diabetes based on diagnostic measurements.

---

## Class Balance

Outcome Distribution:

* Non-Diabetic (0): 500 patients
* Diabetic (1): 268 patients

Observation:

The dataset is moderately imbalanced, with more non-diabetic cases than diabetic cases.

---

## Missing Value Strategy

Several features contained invalid zero values that represent missing measurements:

* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI

Strategy Used:

* Replaced invalid zero values with the median of the respective feature.
* Median imputation was chosen because it is less sensitive to outliers than the mean.

---

## Feature Distributions

Feature distributions were analyzed using histograms and summary statistics.

Observations:

* Glucose values show a wide spread and are strongly associated with diabetes outcome.
* BMI values are concentrated around the overweight and obese ranges.
* Age distribution is skewed toward younger and middle-aged patients.
* Insulin values exhibit high variability and contain several extreme values.

---

## Key Insights

### 1. Glucose is the Strongest Predictor

Patients with higher glucose levels are more likely to be diabetic.

### 2. BMI and Age Influence Diabetes Risk

Higher BMI and older age groups tend to show increased diabetes prevalence.

### 3. Class Imbalance Must Be Considered

The dataset contains more non-diabetic than diabetic cases, making evaluation metrics such as Precision, Recall, F1-Score, and ROC-AUC important in addition to Accuracy.
