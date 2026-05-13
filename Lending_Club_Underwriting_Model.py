# ==============================================================================
# Phase 0 & 1: Environment Setup & Safe Data Loading
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier
import shap
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("1. Locating and Loading Lending Club Data...")

data_path = ""

for root, dirs, files in os.walk('/kaggle/input/'):
    for file in files:
        if "accepted" in file.lower() and (file.endswith(".csv") or file.endswith(".csv.gz")):
            data_path = os.path.join(root, file)
            break
    if data_path:
        break

if not data_path:
    raise FileNotFoundError("Dataset not found. Please check the Kaggle input path.")

print(f"✅ Data path successfully locked: {data_path}")

# 🌟 Memory-Safe Loading: Use chunksize to randomly sample 150,000 records (sufficient for MVP)
print("Sampling 150,000 records to prevent memory overflow...")
chunk_size = 100000
chunks = []
# Set a fixed random seed to ensure consistent sampling results
np.random.seed(42) 

for chunk in pd.read_csv(data_path, chunksize=chunk_size, low_memory=False):
    # Randomly sample 7% of data from each chunk (totaling ~150k-160k records)
    sampled_chunk = chunk.sample(frac=0.07, random_state=42)
    chunks.append(sampled_chunk)

df = pd.concat(chunks, ignore_index=True)

# 1. Define Target (Keep only completed loans)
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default'])]
df['TARGET'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)

print(f"✅ Data Loading Complete! MVP Dataset shape: {df.shape}")
print(f"📊 Default Rate in sample: {df['TARGET'].mean():.2%}")

# ==============================================================================
# Phase 2: Anti-Leakage & Feature Engineering (The Professional Way)
# ==============================================================================
print("\n2. Engineering Features and Dropping Data Leakage...")

# [Core Business Logic]: Drop 'future data' generated after loan origination (Data Leakage)
# [Ultimate Anti-Leakage List] Added last pulled FICO scores, settlement status, and credit pull dates
leakage_cols = [
    'recoveries', 'collection_recovery_fee', 'total_pymnt', 'total_pymnt_inv', 
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'last_pymnt_d', 
    'last_pymnt_amnt', 'next_pymnt_d', 'out_prncp', 'out_prncp_inv', 
    'settlement_status', 'settlement_date', 'settlement_amount', 'hardship_flag',
    'loan_status', 'last_fico_range_high', 'last_fico_range_low', 
    'debt_settlement_flag'
]

# Additionally drop all columns related to 'last_credit_pull' (as get_dummies expands them)
df = df.loc[:, ~df.columns.str.contains('last_credit_pull')]
df = df.drop(columns=[col for col in leakage_cols if col in df.columns], errors='ignore')

# Drop ID-type and complex raw text columns with no predictive value
drop_text_cols = ['id', 'member_id', 'url', 'desc', 'emp_title', 'title', 'zip_code', 'issue_d', 'earliest_cr_line']
df = df.drop(columns=[c for c in drop_text_cols if c in df.columns], errors='ignore')

# Business Feature Engineering: Clean employment length (emp_length) e.g., "10+ years" -> 10
if 'emp_length' in df.columns:
    df['emp_length'] = df['emp_length'].fillna('0')
    df['emp_length_num'] = df['emp_length'].str.extract(r'(\d+)').astype(float)
    df['emp_length_num'] = df['emp_length_num'].fillna(0)
    df = df.drop(columns=['emp_length'])

# Business Feature Engineering: Clean loan term (term) e.g., " 36 months" -> 36
if 'term' in df.columns:
    df['term_months'] = df['term'].str.extract(r'(\d+)').astype(float)
    df = df.drop(columns=['term'])

# Build strong risk combination features (Leverage Ratios)
df['CREDIT_INCOME_RATIO'] = df['loan_amnt'] / (df['annual_inc'] + 1)
df['INSTALLMENT_INCOME_RATIO'] = (df['installment'] * 12) / (df['annual_inc'] + 1)

# ==============================================================================
# Phase 3: Preprocessing
# ==============================================================================
print("\n3. Preprocessing (Handling missing values and encoding)...")

# Drop columns with over 50% missing values (many Lending Club features are highly sparse)
missing_pct = df.isnull().mean()
cols_to_keep = missing_pct[missing_pct < 0.5].index
df = df[cols_to_keep]

y = df['TARGET']
X = df.drop(columns=['TARGET'])

# One-Hot Encoding
X = pd.get_dummies(X, dummy_na=True)

# Handle special JSON/character column name issues for XGBoost compatibility
import re
X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==============================================================================
# Phase 4: XGBoost Modeling
# ==============================================================================
print("\n4. Training XGBoost Model...")

# Dynamically calculate sample weights to handle default class imbalance
scale = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    scale_pos_weight=scale,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_preds = xgb_model.predict(X_test)

print("\n--- Model Validation Metrics ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, xgb_probs):.4f}")
print("\nClassification Report (Focus on Class 1 Recall):")
print(classification_report(y_test, xgb_preds))

# ==============================================================================
# Phase 5: Business Risk Segmentation & SHAP
# ==============================================================================
print("\n5. Generating Business Risk Segmentation...")

eval_df = pd.DataFrame({'Actual_Default': y_test, 'Pred_Probability': xgb_probs})
eval_df['Ranked_Prob'] = eval_df['Pred_Probability'].rank(method='first')
eval_df['Risk_Band'] = pd.qcut(eval_df['Ranked_Prob'], q=10, labels=range(1, 11))

risk_report = eval_df.groupby('Risk_Band').agg(
    Customer_Count=('Actual_Default', 'count'),
    Actual_Defaults=('Actual_Default', 'sum'),
    Min_Prob=('Pred_Probability', 'min'),
    Max_Prob=('Pred_Probability', 'max')
).reset_index()

risk_report['Actual_Default_Rate'] = (risk_report['Actual_Defaults'] / risk_report['Customer_Count']).map("{:.2%}".format)

def assign_action(band):
    if band <= 4: return 'Auto-Approve (Low Risk)'
    elif band <= 8: return 'Manual Review (Medium Risk)'
    else: return 'Auto-Reject (High Risk)'

risk_report['Action'] = risk_report['Risk_Band'].apply(assign_action)
display(risk_report)

print("\nGenerating SHAP Explanations (Takes a few seconds)...")
explainer = shap.TreeExplainer(xgb_model)

# Sample 2000 records for plotting to speed up execution
X_test_sample = X_test.sample(2000, random_state=42)
shap_values = explainer.shap_values(X_test_sample)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, max_display=15, show=False)
plt.title('Global Feature Explainability (SHAP Value)', fontweight='bold')
plt.tight_layout()
plt.show()

print("\n🎉 Lending Club Credit Risk Pipeline Completed!")
