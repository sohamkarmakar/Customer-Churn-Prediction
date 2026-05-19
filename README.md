# Customer Churn Prediction using Machine Learning 🔮

An end-to-end, production-grade Machine Learning classification system designed to analyze and predict customer churn for a telecommunication subscriber base. 

This repository provides a complete, production-grade Machine Learning classification system designed to analyze subscriber behaviors, predict churn risks, and provide actionable recommendations.

---

## 📂 Project Directory Structure

The project is structured following industry-standard modular software engineering practices:

```
customer-churn-prediction/ (workspace root)
│
├── data/
│   ├── raw/                  # Original raw dataset from Kaggle
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed/            # Cleaned, engineered, and split CSVs
│       ├── churn_train.csv
│       └── churn_test.csv
│
├── notebooks/
│   └── customer_churn_eda_modeling.ipynb  # Step-by-step interactive Jupyter Notebook
│
├── src/                      # Modular Python Codebase
│   ├── __init__.py
│   ├── preprocess.py         # Data cleaning, encoding, scaling, and feature engineering
│   ├── plots.py              # Visual plot generator for EDA, comparison, and importances
│   ├── train.py              # Model baseline fitting, hyperparameter tuning, and cross-validation
│   └── pipeline.py           # Class wrapper for real-time model inference
│
├── models/
│   └── best_model.pkl        # Serialized best model + preprocessor pipeline bundle
│
├── outputs/                  # Static execution assets
│   ├── plots/                # High-resolution (300 DPI) EDA figures
│   ├── confusion_matrices/   # Model confusion matrix plots
│   └── model_comparison.csv  # Final performance comparison CSV
│
├── app/
│   └── app.py                # Premium Streamlit web application frontend
│
├── requirements.txt          # Third-party library dependencies
├── README.md                 # Project handbook (this file)
└── main.py                   # Master orchestration terminal CLI
```

---

## 📈 Business Problem Statement

Customer churn occurs when a customer cancels their active subscription with the provider. For telecom businesses, acquiring new customers costs **5 to 25 times more** than retaining existing ones. 

### Financial Impact Model:
- Let's assume a customer contributes a monthly billing average of **$65**.
- If we lose **100 customers per month**, that results in **$6,500/month** or **$78,000/year** of lost Monthly Recurring Revenue (MRR).
- By deploying this prediction pipeline, the business can accurately flag high-risk customers, offer them targeted retention discounts (e.g., $10 off or contract upgrades), saving **80%** of those customers at a fraction of the acquisition cost.

> [!IMPORTANT]
> **Recall is our Priority Metric!**
> - **False Negative (We predict customer won't churn, but they do)**: Total Loss of Customer MRR + High Acquisition Cost. (Extremely Expensive!)
> - **False Positive (We predict customer will churn, but they stay)**: Small expense of a preventative promotional coupon. (Negligible Cost.)
> - Therefore, we must pick a model that maximizes **Recall (Sensitivity)** and **F1-Score** over pure raw Accuracy.

---

## 📊 Dataset Specifications
The project utilizes the classic **Kaggle Telco Customer Churn dataset** (7,043 rows, 21 columns).

### Primary Features Evaluated:
- **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Services Subscribed**: `PhoneService`, `MultipleLines`, `InternetService` (DSL, Fiber optic), `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Account & Financials**: `tenure` (months active), `Contract` (Month-to-month, One year, Two year), `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
- **Target Variable**: `Churn` (Yes / No)

---

## 🛠️ Feature Engineering & Preprocessing
To deliver high-performance predictive analytics, the following feature engineering steps were implemented in [src/preprocess.py](src/preprocess.py):
1. **`tenure_group`**: Segmented numeric tenure into business cycles (`0-1 Year`, `1-2 Years`, `2-4 Years`, `4-5 Years`, `Over 5 Years`).
2. **`Number_of_Services`**: Combined sum of active auxiliary online services used by the subscriber.
3. **`Has_Partner_and_Dependents`**: Interaction variable highlighting strong family binding which statistically decreases churn.
4. **`Monthly_to_Total_Ratio`**: Ratio showcasing if a customer's monthly recurring expense is disproportionately high compared to their historical spend.
5. **Data Leakage Safeguard**: Features scaled using `StandardScaler` and encoded via `OneHotEncoder` *only* after splitting training and testing data.

---

## ⚙️ Installation & Running Steps

### 1. Prerequisite Installations
Ensure Python 3.8+ is installed on your system. Navigate to the project root and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execution Options (Using CLI `main.py`)
Run the master orchestrator script based on your requirements:

- **Option A: Run Full Pipeline (EDA + Preprocessing + Training + Serialization)**
  ```bash
  python main.py --all
  ```
  *This automatically preprocesses the raw CSV, creates processed train/test partitions, generates 7 professional EDA plots, fits four machine learning models, tunes the best model using GridSearchCV, generates comparison metrics, and outputs the serialized model pickle to `models/best_model.pkl`.*

- **Option B: Run only visual EDA generation**
  ```bash
  python main.py --eda
  ```

- **Option C: Run only model training & validation**
  ```bash
  python main.py --train
  ```

### 3. Launching the Interactive Web Application
Run the Streamlit frontend locally:
```bash
streamlit run app/app.py
```
*This starts a server and launches your browser at `http://localhost:8501`. Input customer details, view the exact churn probability, check risk bands, and view the actionable customer playbook!*

---

## 📊 Machine Learning Model Comparison Results

Below is the summary of models trained on the Telco dataset (Stratified 80-20 Split):

| Machine Learning Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 74.9% | 52.4% | 80.2% | 63.4% | 83.9% |
| **Decision Tree** | 78.5% | 58.7% | 69.4% | 63.6% | 82.2% |
| **Random Forest** | 78.2% | 58.1% | 71.7% | 64.2% | 83.8% |
| **XGBoost (Tuned Ensemble)** | **80.4%** | **61.9%** | **78.6%** | **69.2%** | **84.8%** |

*Note: XGBoost tuned with GridSearchCV consistently secures the highest F1-Score (69.2%) and ROC-AUC (84.8%), balancing high recall (78.6%) and precision (61.9%) to prevent false retention giveaways while catching the highest percentage of churning customers.*

---

## ⚙️ Key Methodological Details

### 1. Handling the `TotalCharges` Data Cleaning Challenge
The dataset contains brand-new customers with a `tenure` of `0` months. For these entries, `TotalCharges` is filled with an empty space string `" "`. This causes Pandas to load the entire column as strings. This was resolved by converting empty spaces into `NaN`, parsing the column to `float`, and filling the missing values with `0.0` (as logical total charges for a new customer must be zero).

### 2. Safeguarding Against Data Imbalance
Since loyal customers outweigh churned ones (~74% to 26%), standard classifiers could overfit to the majority class. This was handled by:
1. Using a **Stratified Train-Test Split** so the exact ratio of churners is preserved in both folds.
2. Setting `class_weight='balanced'` in Random Forest and Logistic Regression models, and computing a precise `scale_pos_weight` in XGBoost. This assigns a higher weight penalty to errors made on the minority class during optimization.
3. Prioritizing **Recall** and **F1-Score** during model selection.

### 3. Preventing Data Leakage
To strictly safeguard the code, the train-test split was performed *before* any transformation operations. The scaling and encoding parameters (using `StandardScaler` and `OneHotEncoder`) were fit *only* on the training split, and then used to transform both the test split and real-time inputs.

---

## 🔮 Future Enhancements
- **SHAP (SHapley Additive exPlanations)**: Integrate a SHAP explainability dashboard into Streamlit to visually interpret feature contributions for individual predictions.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Experiment with generating artificial minority samples to evaluate if F1-scores improve.
- **CICD pipeline**: Setup GitHub Actions to run sanity unit tests on the preprocessing functions upon every push.
