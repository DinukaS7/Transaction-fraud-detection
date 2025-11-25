# 🛡️ Transaction Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange)
![Azure](https://img.shields.io/badge/Deployment-Azure-0078D4)

## 📋 Project Overview
This project is a Machine Learning-powered web application designed to detect fraudulent financial transactions. Using a **Logistic Regression** model trained on a massive dataset of financial logs, the app can predict in real-time whether a specific transaction (Payment, Transfer, Cash Out, etc.) is legitimate or fraudulent.

The application is built with **Streamlit** for the frontend and deployed on **Microsoft Azure**.

## 📊 Exploratory Data Analysis (EDA)
Extensive data analysis was performed to identify key indicators of fraud.

### 1. Transaction Landscape
We analyzed the volume of different transaction types and their respective fraud rates.
| Transaction Volume | Fraud Rates |
| :---: | :---: |
| ![Transaction Types](assets/Transaction%20Types.png) | ![Fraud Rate](assets/Fraud%20Rate%20by%20Type.png) |

### 2. Fraud Identification
Crucially, we found that fraud occurs **exclusively** in `TRANSFER` and `CASH_OUT` transaction types.
![Fraud Distribution](assets/Fraud%20Distributio%20in%20Transfer%20&%20Cash_Out.png)

### 3. Financial Analysis
We analyzed the distribution of transaction amounts. By applying a Log Scale transformation, we normalized the data. Boxplots reveal that fraudulent transactions tend to occur at specific higher value ranges compared to normal transactions.

| Distribution (Log Scale) | Amount vs Fraud Class |
| :---: | :---: |
| ![Distribution](assets/Transaction%20Amount%20Distribution.png) | ![Amount vs Fraud](assets/Amount%20vs%20isFraud.png) |

### 4. Temporal & Feature Correlation
Analyzing frauds over time steps showed consistent attack patterns. The heatmap reveals strong correlations between `oldbalanceOrg` and `newbalanceOrig`, which are critical indicators for account draining schemes.

| Time Analysis | Feature Correlation |
| :---: | :---: |
| ![Time](assets/Frauds%20Over%20Time.png) | ![Correlation](assets/Correlation%20Matrix.png) |

---

## 🤖 Model Architecture
The model was built using `scikit-learn` Pipelines to ensure clean preprocessing and inference.

* **Preprocessing:**
    * **Categorical Features:** OneHotEncoding for `type`.
    * **Numerical Features:** StandardScaler for `amount` and balance columns.
* **Algorithm:** Logistic Regression
* **Parameters:** `class_weight='balanced'` (To handle the severe class imbalance between fraud/non-fraud).
* **Performance:** The model achieves high recall, crucial for fraud detection to minimize missed fraudulent cases.

```python
# Model Pipeline Snippet
pipeline = Pipeline([
    ("prep", ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ])),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])
