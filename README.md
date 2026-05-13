# Lending Club Credit Risk Modeling with XGBoost and SHAP

## Description

This project builds a credit risk modeling pipeline using Lending Club loan data. The goal is to predict whether a completed loan is likely to become a default or charged-off loan.

The project applies data loading, anti-leakage feature selection, feature engineering, preprocessing, XGBoost classification, business risk segmentation, and SHAP explainability. It is designed as a portfolio project for credit risk analytics, machine learning, and business decision support.

## Dataset

This project uses the Lending Club loan dataset from Kaggle:

**All Lending Club Loan Data by wordsforthewise**

Due to file size and licensing considerations, the raw dataset is not included in this repository. Users can download the dataset from Kaggle and run the notebook in Kaggle, or place the data file in a local `data/` folder.

## Project Objectives

- Predict loan default risk using historical Lending Club loan data
- Avoid data leakage by removing post-origination repayment and recovery fields
- Handle class imbalance using XGBoost sample weighting
- Segment loans into business risk bands
- Use SHAP values to explain model predictions
- Translate model outputs into practical credit decision actions

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- matplotlib
- seaborn
- Kaggle Notebook

## Project Structure

```text
lending-club-credit-risk-modeling/
├── Lending_Club_Credit_Risk_Modeling.ipynb
├── README.md
├── risk_band_report.csv
└── shap_summary.png
```

## Methodology

### 1. Data Loading

The notebook dynamically searches the Kaggle input directory for the Lending Club accepted loan dataset. To reduce memory usage, the dataset is loaded in chunks and randomly sampled.

### 2. Target Definition

Only completed loans are used for modeling. The target variable is defined as:

- `0`: Fully Paid
- `1`: Charged Off or Default

### 3. Data Leakage Prevention

Post-origination fields such as payment amounts, recoveries, last payment dates, settlement information, and last FICO scores are removed to reduce data leakage.

Examples of removed leakage columns include:

```text
total_pymnt
recoveries
collection_recovery_fee
last_pymnt_d
last_pymnt_amnt
settlement_status
last_fico_range_high
last_fico_range_low
loan_status
```

### 4. Feature Engineering

The project creates additional risk-related features, including:

```text
CREDIT_INCOME_RATIO
INSTALLMENT_INCOME_RATIO
emp_length_num
term_months
```

These features help represent borrower leverage and repayment burden.

### 5. Model Training

An XGBoost classifier is trained to predict loan default risk. Class imbalance is handled using `scale_pos_weight`.

### 6. Model Evaluation

The model is evaluated using:

- ROC-AUC score
- Classification report
- Precision
- Recall
- F1-score

The focus is on identifying higher-risk loans while maintaining interpretable business outputs.

### 7. Business Risk Segmentation

Predicted default probabilities are divided into 10 risk bands. Each band is mapped to a business action:

| Risk Band | Action |
|---|---|
| 1–4 | Auto-Approve |
| 5–8 | Manual Review |
| 9–10 | Auto-Reject |

### 8. Model Explainability

SHAP is used to explain the global feature importance of the XGBoost model.

## Sample Output

### SHAP Summary Plot

![SHAP Summary](shap_summary.png)

### Risk Band Report

The model ranks borrowers into risk bands based on predicted default probability. Higher bands represent higher predicted credit risk.

The risk band report is saved as:

```text
outputs/risk_band_report.csv
```

## Key Takeaways

- Removing data leakage is critical in credit risk modeling.
- XGBoost can effectively model nonlinear relationships in borrower and loan attributes.
- Risk segmentation makes model output easier to translate into business decisions.
- SHAP improves model transparency by showing which features contribute most to risk prediction.

## Limitations

This project is for educational and portfolio purposes. It uses historical Lending Club data and does not represent a production credit approval system. Real-world credit models require additional validation, fairness testing, regulatory review, monitoring, and business policy constraints.

## Future Improvements

- Add time-based train/test split using loan issue dates
- Compare XGBoost with Logistic Regression and Random Forest
- Add model calibration for probability reliability
- Add fairness analysis across borrower groups
- Build an interactive dashboard for risk band monitoring
- Deploy the model as a simple API or web app

