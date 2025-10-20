## 🎯 Context and Aim

This module is part of a binary classification project aimed at predicting customer churn. In the previous session, a logistic regression model was trained and evaluated using accuracy, which reached 80%. However, this metric alone is insufficient, especially with imbalanced datasets where the majority of customers do not churn.

The goal of this module is to **critically assess model performance** using more nuanced and robust evaluation metrics. Specifically, it introduces:

* The **confusion matrix** to categorize prediction types (TP, TN, FP, FN),
* **Precision** and **recall** to understand the trade-offs between prediction accuracy and completeness,
* The **ROC curve** and **AUC** to measure the model's ability to distinguish classes across thresholds,
* And **cross-validation** to estimate performance stability and guide hyperparameter tuning.

This ensures the model is not just accurate on paper, but truly effective in identifying churning customers in practice.

Sure! Here’s a clearer and more digestible version of the **Global Summary** — still concise, but easier to reread quickly:

---

## 🧭 Global Summary

* 📊 We started by questioning the meaning of **80% accuracy** — is it really good?
* 📦 Introduced the **confusion matrix** to split predictions into:

  * True/False Positives (TP, FP)
  * True/False Negatives (TN, FN)
* 🎯 **Precision**: Of the customers we predicted would churn, how many really did?
* 📢 **Recall**: Of all customers who churned, how many did we catch?
* 📉 **ROC Curve**: Showed how performance changes depending on the threshold.
* 🧮 **AUC** (Area Under Curve): Summarized how well the model separates churners vs. non-churners.
* 🔁 **Cross-validation**: Split data multiple times to check how stable and reliable the model is.
* ⚙️ We tuned the **C parameter** (regularization) to improve generalization.

📌 **Takeaway**: Don’t rely only on accuracy — use multiple metrics to understand real performance.

## 📺 4.1 – Evaluation Metrics Session Overview

### ✅ Goal

Understand how to evaluate binary classification models beyond just accuracy.

### 🧠 Concept

Accuracy can be misleading, especially with class imbalance. The session introduces confusion matrix, precision, recall, ROC curve, AUC, and cross-validation.

### 🛠️ Tasks

* Recall logistic regression model trained for churn prediction.
* Highlight limitations of accuracy.
* Introduce key metrics for evaluating classifiers.


Perfect — here’s the structured breakdown for the **missing session** in the expected format:

---

# 🧩 Session-by-Session Breakdown

## 📺 4.2 – Accuracy and Dummy Model

### ✅ Goal

Evaluate model accuracy across different classification thresholds and compare it to a simple dummy model baseline.

### 🧠 Concept

Accuracy alone can be misleading in imbalanced datasets. A model that always predicts the majority class (e.g. “no churn”) can still appear to perform well. By sweeping through different thresholds and comparing with a dummy model, we get a better sense of how meaningful the model’s performance actually is.

### 🛠️ Tasks

* Compute accuracy at the default threshold (0.5).
* Evaluate accuracy for multiple thresholds from 0.0 to 1.0.
* Visualize accuracy as a function of the threshold.
* Compare model accuracy to a **dummy classifier** that always predicts “no churn.”
* Highlight limitations of accuracy with class-imbalanced data.

### 💻 Microcode

```python
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Accuracy at threshold = 0.5
accuracy_score(y_val, y_pred >= 0.5)

# Evaluate accuracy across thresholds
thresholds = np.linspace(0, 1, 21)
scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)

# Plot accuracy vs threshold
plt.plot(thresholds, scores)

# Check prediction behavior at extreme threshold
Counter(y_pred >= 1.0)

# Accuracy of dummy model that predicts all "no churn"
1 - y_val.mean()
```

Let me know if you want to integrate this into a full document or summary file!

---

## 📺 4.3 – Confusion Matrix

### ✅ Goal

Analyze model errors using the confusion matrix.

### 🧠 Concept

Confusion matrix shows four types of predictions: TP, TN, FP, FN, helping understand not just *how many* were correct, but *how*.

### 🛠️ Tasks

* Define prediction thresholds.
* Calculate confusion matrix elements.
* Visualize and interpret errors.

### 💻 Microcode

```python
# Threshold-based binary prediction
t = 0.5
predict_positive = y_pred >= t
actual_positive = y_val == 1
true_positive = (predict_positive & actual_positive).sum()
false_positive = (predict_positive & ~actual_positive).sum()
false_negative = (~predict_positive & actual_positive).sum()
true_negative = (~predict_positive & ~actual_positive).sum()
```

---

## 📺 4.4 – Precision and Recall


### ✅ **Goal**

Measure how well the model handles **positive predictions** (churns):

* **Are they accurate?** → Precision
* **Are they complete?** → Recall

---

### 🧠 **Key Concepts**

#### 🎯 **Precision** — *How correct are your alerts?*

> Of all customers **predicted as churners**, how many truly were?

* High precision = few false alarms
* Focuses on **quality** of positive predictions

**Formula:**

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives + False Positives}}
$$

---

#### 🔍 **Recall** — *How many did you catch?*

> Of all customers who **actually churned**, how many did we correctly detect?

* High recall = few missed churners
* Focuses on **coverage** of the positive class

**Formula:**

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}
$$

---

### 🧪 Example

If:

* 210 customers were correctly predicted as churners (**TP**)
* 101 were wrongly flagged as churners (**FP**)
* 176 churners were missed entirely (**FN**)

Then:

* **Precision** = $ \frac{210}{210 + 101} = 67.5%$
* **Recall** = $ \frac{210}{210 + 176} = 54.4%$

---

### 🧠 Interpretation:

| Metric    | Focus             | High Value Means                 |
| --------- | ----------------- | -------------------------------- |
| Precision | *False alarms*    | Few non-churners wrongly flagged |
| Recall    | *Missed churners* | Few churners go undetected       |


### 🛠️ Tasks

* Visualize precision and recall.
* Compute formulas from confusion matrix.

### 💻 Microcode

```python
precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
```

---

Here’s a clearer and more informative version of your ROC Curve explanation:

---

## 📺 **4.5 – ROC Curve**

### ✅ **Goal**

Evaluate how well the model **distinguishes** between classes **across all thresholds** — not just at 0.5.

---

### 🧠 **Concept**

The **ROC Curve** (Receiver Operating Characteristic) shows the trade-off between:

* **True Positive Rate (TPR)** = **Recall**
  → How many actual positives (churners) were correctly identified.

* **False Positive Rate (FPR)**
  → How many actual negatives (non-churners) were incorrectly flagged as positive.

As the **decision threshold** varies from 0 → 1:

* TPR and FPR change
* The ROC curve traces this evolution

---

### 📈 **Axes:**

* **X-axis**: False Positive Rate = $\frac{FP}{FP + TN}$
* **Y-axis**: True Positive Rate = $\frac{TP}{TP + FN}$

---

### ✅ **Why It’s Useful**

* Helps **compare classifiers** regardless of chosen threshold.
* Good for **imbalanced datasets**, since it considers both error types.
* A model with no skill lies on the **diagonal** (TPR = FPR).
* The **closer to the top-left**, the **better** the model.

---

### 🛠️ Tasks

* Compute TPR and FPR at multiple thresholds.
* Compare with dummy and ideal models.
* Plot ROC curves.

### 💻 Microcode

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_val, y_pred)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
```

---

## 📺 4.6 – ROC AUC

### ✅ Goal

Quantify model performance using AUC.

### 🧠 Concept

AUC = probability that a randomly selected positive has a higher score than a negative.

### 🛠️ Tasks

* Implement AUC manually and with sklearn.
* Understand interpretation via score distributions.

### 💻 Microcode

```python
from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_val, y_pred)
print(f"AUC = {auc:.3f}")
```

---

## 📺 4.7 – Cross-Validation

### ✅ Goal

Evaluate model robustness and tune hyperparameters.

### 🧠 Concept

Split the data multiple times to measure model stability and avoid overfitting.

### 🛠️ Tasks

* Define `train()` and `predict()` functions.
* Use `KFold` to compute scores across multiple folds.
* Tune hyperparameter `C` of logistic regression.

### 💻 Microcode

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []

for train_idx, val_idx in kf.split(df_train):
    df_tr, df_val = df_train.iloc[train_idx], df_train.iloc[val_idx]
    dv, model = train(df_tr)
    y_pred = predict(df_val, dv, model)
    auc = roc_auc_score(df_val.churn, y_pred)
    scores.append(auc)

print(np.mean(scores), np.std(scores))
```

---

## 📺 4.8 – Summary

### ✅ Goal

Recap all evaluation metrics covered and clarify when to use each.

### 🧠 Concept

Each metric offers a different perspective:

* Accuracy: general correctness
* Precision: correctness among positives
* Recall: completeness of positives
* AUC: overall separability
* CV: robustness and parameter tuning

### 🛠️ Tasks

* Review when to prefer each metric.
* Highlight model limitations via metrics.
* Prepare for deployment in next session.

---

Let me know if you’d like this exported as Markdown or integrated into your ML Zoomcamp notes.
