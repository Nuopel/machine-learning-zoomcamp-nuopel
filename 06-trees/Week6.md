# 📘 Credit Risk Scoring Project (ML Zoomcamp Session 6)

## 🎯 Context and Aim

This project aims to build a predictive model for credit risk scoring using decision trees and ensemble learning techniques. In a typical banking context, when a client applies for a loan (e.g., to buy a phone), the bank must assess whether to grant or deny the loan. The goal of this project is to train machine learning models that estimate the **probability of default** (risk that a customer will not repay) based on historical data. This is a **binary classification problem** where the output variable indicates either *default* or *no default*. The dataset used, `credit_scoring.csv`, contains various features like income, expenses, loan amount, and categorical variables (homeownership, marital status, etc.).

# 🥩 Session-by-Session Breakdown

## 📺 Session 6.1 – Project Introduction

### ✅ Goal

Introduce the credit scoring context and explain how machine learning can support risk-based lending decisions.

### 🧠 Concept

Binary classification is used to predict whether a customer will default. We aim to estimate \$P(\text{default} | x)\$ given historical features \$x\$. Historical data includes features like income, homeownership, loan amount, and a target variable "status".

### 🛠️ Tasks

* Understand business context of credit risk
* Define the prediction target
* Outline project steps: dataset exploration, preprocessing, model training, evaluation

### 💻 Microcode

```python
# Goal: Predict default probability
# y = 1 if default, y = 0 otherwise
# X = input features (e.g., income, assets, loan duration, etc.)
```

## 📺 Session 6.2 – Data Cleaning and Preparation

### ✅ Goal

Preprocess the dataset to make it suitable for training ML models.

### 🧠 Concept

Raw data may have:

* Encoded categorical variables (e.g., 1 = rent, 2 = owner)
* Missing values (e.g., 999999 for income)
* Non-standardized column names
  Cleaning includes decoding categorical values, treating missing data, filtering, and splitting into train/val/test sets.

### 🛠️ Tasks

* Lowercase column names
* Decode categorical variables using `map`
* Replace numeric codes for missing values with `np.nan`
* Filter unknown target entries
* Train/validation/test split
* Convert target to binary: default (1) vs. non-default (0)

### 💻 Microcode

```python
# Decode status
status_map = {1: 'ok', 2: 'default', 0: 'unknown'}
df['status'] = df['status'].map(status_map)

# Replace missing values (e.g. 999999)
for col in ['income', 'assets', 'debt']:
    df[col] = df[col].replace(99999999, np.nan)

# Filter only known status
filtered_df = df[df.status != 'unknown'].reset_index(drop=True)

# Binary target
y_train = (df_train.status == 'default').astype(int)
```

## 📺 Session 6.3 – Decision Trees

### ✅ Goal

Train a simple decision tree classifier and evaluate its performance.

### 🧠 Concept

Decision trees learn rules in the form of if/else logic based on feature thresholds. The tree recursively splits the dataset until leaves predict a target value.
Risk: if the tree is too deep, it **overfits** the training data.

### 🛠️ Tasks

* Use `DictVectorizer` to transform data
* Train `DecisionTreeClassifier`
* Evaluate AUC on train and validation sets
* Visualize tree rules

### 💻 Microcode

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

# Train tree
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_train.to_dict(orient='records'))
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Predict and evaluate
X_val = dv.transform(df_val.to_dict(orient='records'))
y_pred = model.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)
```

## 📺 Session 6.4 – Decision Tree Learning Algorithm

### ✅ Goal

Understand how decision trees choose the best splits during training.

### 🧠 Concept

At each node, the algorithm tries all possible thresholds for all features and selects the one minimizing impurity (e.g., Gini, entropy, misclassification rate).

### 🛠️ Tasks

* Simulate split selection on toy dataset
* Evaluate impurity of each possible threshold
* Choose split with lowest weighted impurity

### 💻 Microcode

```python
# Evaluate impurity for all thresholds
best_score = 1.0
for threshold in thresholds:
    left = df[df['feature'] <= threshold]
    right = df[df['feature'] > threshold]
    score = weighted_impurity(left, right)
    if score < best_score:
        best_threshold = threshold
        best_score = score
```

---


### 6.6 Random Forest

🔁 **From Single Tree to Ensemble Learning**

While 6.5 showed how hyperparameter tuning can balance depth and generalization for a single decision tree, Random Forest goes a step further by building many such trees and averaging their predictions. This ensemble method improves **robustness**, **stability**, and **generalization**.

This transition represents a shift:

* **From focusing on one well-tuned tree**,
* **To building many simpler trees trained on slightly different data and features**.

By introducing randomness in both **rows (bootstrap samples)** and **columns (random feature subsets)**, Random Forest reduces variance and avoids the fragility of individual trees — even if that one tree seemed “good enough.”

---

| Aspect               | **6.5 – Single Decision Tree**      | **6.6 – Random Forest Ensemble**        |
| -------------------- | ----------------------------------- | --------------------------------------- |
| **Goal**             | Tune one tree to reduce overfitting | Combine many trees for stability        |
| **Model Type**       | One interpretable tree              | An ensemble of randomized trees         |
| **Training Data**    | Full data used for one tree         | Bootstrap samples (rows + features)     |
| **Variance**         | High (sensitive to noise)           | Low (averaged predictions)              |
| **Bias**             | Can be very low (deep trees)        | Slightly higher, but generalizes better |
| **Overfitting Risk** | High if tree is too deep            | Low, even with many trees               |
| **Interpretability** | Very high (tree is readable)        | Lower (individual trees are hidden)     |

---

🎯 **Goal**
Use ensembling (Random Forest) to improve prediction stability and performance over a single decision tree.

🧠 **Concept**
Random Forest builds multiple decision trees on bootstrapped samples of the data and aggregates their predictions. This ensemble approach reduces variance and overfitting, while maintaining good interpretability. The randomness comes from sampling rows and subsets of features.

🛠️ **Tasks**

* Train Random Forest with increasing number of trees (`n_estimators`)
* Tune `max_depth` and `min_samples_leaf` jointly
* Plot AUC as function of tree count and depth
* Evaluate stability of model performance

💡 **Note**
Random Forest tends to outperform a single tree without requiring heavy tuning. It’s less prone to overfitting, especially with many trees.



### 6.7 Gradient Boosting and XGBoost

🎯 **Goal**
Introduce gradient boosting and train the first XGBoost model, observing its stepwise training process.

🧠 **Concept**
Gradient boosting builds trees sequentially, where each tree attempts to correct the mistakes of the previous one. XGBoost is an optimized library for this approach. It’s powerful but sensitive to hyperparameters and prone to overfitting.

🛠️ **Tasks**

* Install and use XGBoost
* Train with a fixed set of hyperparameters
* Monitor AUC at each boosting round
* Parse and visualize training curves (train vs validation AUC)

💡 **Note**: Boosting is very effective on structured/tabular data, often outperforming Random Forest. However, it needs careful tuning (e.g., learning rate, tree depth).

---

### 6.8 XGBoost Parameter Tuning

🎯 **Goal**
Fine-tune XGBoost to maximize performance on validation data without overfitting.

🧠 **Concept**
XGBoost has many hyperparameters; tuning the learning rate (`eta`), depth (`max_depth`), and minimum weight (`min_child_weight`) can greatly affect performance. A small `eta` with many rounds allows more precise adjustments. Larger `min_child_weight` values act as regularizers.

🛠️ **Tasks**

* Grid search over combinations of `eta`, `max_depth`, `min_child_weight`
* Plot AUC as function of boosting round
* Select optimal configuration based on validation AUC curves

💡 **Note**: Boosting with high `eta` may overfit quickly; low `eta` leads to better generalization but slower convergence. Tuning is crucial for robust performance.


## 📺 Session 6.9 – Selecting the Best Model

### ✅ Goal

Compare trained models (Decision Tree, Random Forest, XGBoost) and select the best based on validation AUC, then evaluate it on the test set.

### 🧠 Concept

XGBoost generally performs best on tabular data but requires careful parameter tuning. Simpler models like Decision Trees or Random Forests are easier to interpret and tune but may underperform.

### 🛠️ Tasks

* Compare validation AUC for the best models
* Retrain final model (XGBoost) on full training data
* Evaluate final model on held-out test set

### 💻 Microcode

```python
# Full training
full_train = df_train.append(df_val).reset_index(drop=True)
y_full_train = (full_train.status == 'default').astype(int)
full_train = full_train.drop('status', axis=1)

# Vectorize
train_dict = full_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(train_dict)

# Prepare test set
test_dict = df_test.drop('status', axis=1).to_dict(orient='records')
X_test = dv.transform(test_dict)
y_test = (df_test.status == 'default').astype(int)

# Train final model
final_model = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=dv.get_feature_names_out())
final_model_test = xgb.DMatrix(X_test, feature_names=dv.get_feature_names_out())
params = {'eta': 0.3, 'max_depth': 6, 'min_child_weight': 1, 'objective': 'binary:logistic', 'seed': 1}
model = xgb.train(params, final_model, num_boost_round=10)

# Predict and evaluate
y_pred = model.predict(final_model_test)
auc = roc_auc_score(y_test, y_pred)
print(f"Final AUC: {auc:.3f}")
```

### ✅ Outcome

XGBoost showed the highest AUC (≈0.83) and generalized well to the test set. Final model performance confirms good generalization and justifies its selection despite higher complexity.

## 📺 Session 6.10 – Recap

### ✅ Goal

Revoir et synthétiser toutes les étapes du projet de scoring de crédit, de la préparation des données à l’évaluation du modèle final.

### 🧠 Concept

Un projet de machine learning réussi repose sur une chaîne complète : compréhension du problème, nettoyage des données, choix des modèles, validation, et évaluation finale sur un jeu de test non vu.

### 🛠️ Résumé des étapes

1. Chargement et nettoyage des données : traitement des valeurs manquantes, des colonnes codées, et des entrées inconnues.
2. Transformation : vectorisation avec `DictVectorizer` pour travailler avec `scikit-learn`.
3. Modélisation : entraînement de plusieurs modèles (arbre de décision, random forest, XGBoost).
4. Évaluation : comparaison par AUC sur validation, sélection du meilleur modèle.
5. Entraînement final : réentraînement sur l’ensemble train+val, test final sur données jamais vues.

### ✅ Outcome

Le pipeline complet fonctionne bien, le modèle final (XGBoost) généralise correctement, et toutes les étapes sont reproductibles. Le projet est prêt à être intégré dans un flux de production ou enrichi par des techniques supplémentaires (feature engineering, calibration, déploiement, etc.).
