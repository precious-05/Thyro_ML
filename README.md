# Machine Learning | Thyroid Cancer Diagnosis Classification

A complete machine learning pipeline built on an encoded thyroid cancer dataset to classify diagnoses as **Benign** or **Malignant**. This notebook covers class imbalance handling, training multiple classifiers, evaluating them using key metrics, and comparing their performance.

---

## Project Overview

This project takes the pre-processed and encoded thyroid cancer dataset and applies supervised machine learning to predict whether a patient's diagnosis is **Malignant (1)** or **Benign (0)**. Four different classification algorithms are trained, evaluated with confusion matrices and classification reports, and compared side-by-side for accuracy.

---

## 📂 Dataset

**File:** `encoded_thyroid_2.csv`

- All categorical features have been label/ordinal encoded (from the EDA notebook)
- **Target Column:** `Diagnosis_Malignant` → 0 = Benign, 1 = Malignant
- **Features (X):** All columns except `Diagnosis_Malignant`

---

## Class Imbalance Check

Before training, the target variable distribution was visualized using a **Seaborn countplot with percentage labels**:

- The dataset showed significant class imbalance — Benign cases were the majority class
- This was addressed during model training using:
  - `class_weight='balanced'` in Decision Tree and Random Forest
  - `scale_pos_weight=2` in XGBoost
  - **SMOTE (Synthetic Minority Oversampling Technique)** was imported for potential oversampling

---

## 🔀 Train-Test Split

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,      # 80% training, 20% testing
    random_state=42,    # reproducibility
    stratify=Y          # preserves class ratio in both splits
)
```

- **Training Set:** 80% of data
- **Testing Set:** 20% of data
- `stratify=Y` ensures class distribution is maintained in both splits
- Target distribution was verified for both train and test sets after splitting

---

## 🧠 Models Trained

### 1. Logistic Regression
- `max_iter=1000` to ensure convergence
- Trained on imbalanced data (before SMOTE)
- **Accuracy: 82.93%**

### 2. Decision Tree
- `class_weight='balanced'` to handle imbalance
- `random_state=42` for reproducibility
- **Accuracy: 71.20%**

### 3. Random Forest
- `n_estimators=100` (100 decision trees)
- `class_weight='balanced'`
- `random_state=42`
- **Accuracy: 82.61%**

### 4. XGBoost
- `scale_pos_weight=2` to penalize misclassification of minority class
- `eval_metric='logloss'`
- `random_state=42`
- **Accuracy: 82.83%**

---

## 📊 Evaluation & Visualizations

Each model was evaluated using the following metrics and visualizations:

### 🔹 Confusion Matrix Heatmaps
A **Seaborn heatmap** was plotted for every model showing:
- True Positives (TP), True Negatives (TN)
- False Positives (FP), False Negatives (FN)
- Annotated with actual counts; axes labeled as Benign (0) and Malignant (1)

| Confusion Matrix | Model |
|---|---|
| Confusion Matrix: Logistic Regression (Before Balancing) | Logistic Regression |
| Confusion Matrix: Decision Tree (Before SMOTE) | Decision Tree |
| Confusion Matrix: XGBoost | XGBoost |
| Confusion Matrix: Random Forest | Random Forest |

### 🔹 Per-Class Metrics Bar Chart (Logistic Regression)
A **grouped Seaborn bar chart** was plotted for Logistic Regression showing:
- **Precision**, **Recall**, and **F1-Score** per class (Benign vs Malignant)
- Helps visualize how well the model performs on each class individually, not just overall

### 🔹 Model Accuracy Comparison Bar Chart
A final **Seaborn bar chart** comparing all four models side by side:
- X-axis: Algorithm names
- Y-axis: Accuracy (0 to 1)
- Accuracy values annotated on top of each bar

---

## 📈 Model Performance Summary

| Algorithm | Accuracy |
|---|---|
| **Logistic Regression** | **82.93%** ✅ Best |
| XGBoost | 82.83% |
| Random Forest | 82.61% |
| Decision Tree | 71.20% |

> **Logistic Regression** achieved the highest accuracy at **82.93%**, followed closely by XGBoost and Random Forest. Decision Tree performed the weakest at 71.20% despite using `class_weight='balanced'`.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Base plotting |
| Seaborn | Confusion matrix heatmaps and bar charts |
| Scikit-learn | Train-test split, Logistic Regression, Decision Tree, Random Forest, metrics |
| XGBoost | Gradient boosting classifier |
| Imbalanced-learn | SMOTE for oversampling |

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/thyroid-cancer-ml.git
   cd thyroid-cancer-ml
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
   ```

3. Make sure `encoded_thyroid_2.csv` is in the project root (output from the EDA + encoding notebook).

4. Open and run the notebook:
   ```bash
   jupyter notebook actual_ml_training.ipynb
   ```

---

## 🔗 Related Notebooks

| Notebook | Description |
|---|---|
| `thyroid_eda.ipynb` | EDA, outlier detection, bivariate & multivariate analysis |
| `actual_ml_training.ipynb` | ML model training & evaluation (this notebook) |

---

## 👤 Author

**Alina Liaquat**
🔗 [LinkedIn](https://www.linkedin.com/in/alina-liaquat-779347325)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
