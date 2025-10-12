
## ML Zoomcamp Week 3: Churn Prediction Project – Summary

The project aims to identify customers that are likely to churn or stop using a service. Each customer has a score associated with the probability of churning. Considering this data, the company would send an email with discounts or other promotions to avoid churning.

The ML strategy applied to approach this problem is **binary classification**, which for one instance (the $i^{th}$ customer), can be expressed as:

$$
g(x_i) = y_i
$$

In the formula, $y\_i$ is the model's prediction and belongs to ${0, 1}$, with 0 being the negative value (no churning), and 1 the positive value (churning). The output corresponds to the likelihood of churning.

In brief, the main idea behind this project is to build a model with historical data from customers and assign a score of the likelihood of churning.


To address this problem, we follow a standard supervised learning pipeline:

1. **Data preprocessing**: We clean the dataset, convert relevant columns to numeric format, and encode the target (`churn`) as binary.
2. **Feature engineering**: Categorical features are one-hot encoded using `DictVectorizer`, and numerical features are analyzed using correlation.
3. **Exploratory data analysis**: We inspect churn rates, risk ratios, mutual information scores, and correlation coefficients to understand feature relevance.
4. **Modeling**: We train a logistic regression classifier to estimate the probability of churn, using the sigmoid function to map inputs to probabilities.
5. **Evaluation**: The model is evaluated on a validation set using accuracy, confusion matrix, and classification report, then retrained on full data and tested.
6. **Deployment-ready format**: The final model is saved for future use, and optional improvements like threshold tuning, calibration, or tree-based models are suggested.

This approach allows us to interpret key drivers of churn while building a probabilistic model suitable for real-world decision-making.

---

### Setup: Categorical and Numerical Feature Lists

```python
# Declare these once to use throughout the notebook
categorical = [
    'gender', 'partner', 'dependents', 'phoneservice', 'multiplelines', 'internetservice',
    'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv',
    'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod'
]

numerical = ['tenure', 'monthlycharges', 'totalcharges']
```

---

### 📺 3.1 – Project Introduction

**Goal:** Predict churn (leaving customers) using binary classification. Score each customer with a probability and take action (e.g., send promotion).

**Concepts:**

* Binary classification problem ($0 = \text{no churn},\ 1 = \text{churn}$)
* Want to identify users with $y = 1$

```python
# 3.1 – Example score output
scores = {
    'Customer A': 0.20,
    'Customer B': 0.35,
    'Customer C': 0.85  # likely to churn
}
```

---

### 📺 3.2 – Data Preparation

**Tasks:**

* Clean column names
* Convert `TotalCharges` to numeric
* Encode `Churn` to binary

```python
# 3.2 – Data cleaning

# Lowercase and replace spaces
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Convert TotalCharges
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

# Encode churn as binary
df.churn = (df.churn == 'yes').astype(int)
```

---

### 📺 3.3 – Validation Framework

**Split strategy:** 60% train, 20% validation, 20% test

```python
# 3.3 – Train/Val/Test split
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values
```

---

### 📺 3.4 – Exploratory Data Analysis (EDA)

**Key metrics:**

* Global churn rate ≈ 27%

```python
# 3.4 – Churn rate
print(df_full_train.churn.value_counts(normalize=True))
print(df_full_train.churn.mean())  # Alternative way
```

> Expected Output: $\sim 0.27$

---

### 📺 3.5 – Churn Rate & Risk Ratio

**Metrics:**

* Group churn rate
* Risk ratio:

  $$
  \text{Risk Ratio} = \frac{\text{Group Churn Rate}}{\text{Global Churn Rate}}
  $$

```python
# 3.5 – Risk ratio calculation
global_churn = df_full_train.churn.mean()
group = df_full_train.groupby('partner')['churn'].agg(['mean', 'count'])
group['diff'] = group['mean'] - global_churn
group['risk'] = group['mean'] / global_churn
```

---

### 📺 3.6 – Mutual Information

**Concept:**
Mutual Information (MI) measures how much knowing one variable (e.g., a feature) reduces uncertainty about another (e.g., churn).
In this context, it tells us how informative each categorical feature is about whether a customer will churn.

**Formula (conceptual):**

$$
\text{MI}(X, Y) = \sum_{x \in X} \sum_{y \in Y} p(x, y) \log \left( \frac{p(x, y)}{p(x)p(y)} \right)
$$

**Tool:** `mutual_info_score` from `sklearn`

```python
# 3.6 – Mutual info for all categorical features
from sklearn.metrics import mutual_info_score

mi = []
for col in categorical:
    score = mutual_info_score(df_full_train[col], df_full_train.churn)
    mi.append((col, score))

mi = pd.DataFrame(mi, columns=['feature', 'mutual_info'])
mi.sort_values(by='mutual_info', ascending=False)
```

---

### 📺 3.7 – Correlation (Numerical Features)

**Tool:** Pearson correlation coefficient

```python
# 3.7 – Correlation matrix
correlation = df_full_train[numerical + ['churn']].corr()
print(correlation['churn'])
```

> Interpretation:

* $ \text{Corr}(\text{Tenure}, \text{Churn}) < 0 $ ⇒ Longer tenure = less likely to churn
* $ \text{Corr}(\text{MonthlyCharges}, \text{Churn}) > 0 $ ⇒ Higher charges = more likely to churn


---

### 📺 3.8 – One-Hot Encoding

**Tool:** `DictVectorizer`

```python
# 3.8 – One-hot encoding with DictVectorizer
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)
df_train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(df_train_dicts)

X_val = dv.transform(df_val[categorical + numerical].to_dict(orient='records'))
```

> `DictVectorizer` auto-handles numeric values and one-hot encodes categoricals.

---

### 📺 3.9 – Logistic Regression (Theory)

**Concept:**
Use the sigmoid function to convert linear model output into a probability.

Given:

* $x \in \mathbb{R}^d$ is the feature vector
* $w \in \mathbb{R}^d$ are the weights
* $w\_0$ is the bias

We compute the score:

$$
z = w_0 + w^\top x
$$

and convert it into a probability using the **sigmoid function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

This output $p = \sigma(z)$ gives the estimated probability that the customer will churn.

```python
# 3.9 – Sigmoid demo
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = w0 + np.dot(w, x)
p = sigmoid(z)
```

---

### 📺 3.10 – Training Logistic Regression

**Goal:**
Fit a logistic regression model on the training data and evaluate its performance on the validation set using:

* Accuracy
* Confusion Matrix
* Classification Report

**Tool:** `LogisticRegression` from `sklearn`

```python
# 3.10 – Model training and accuracy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
probs = model.predict_proba(X_val)[:, 1]  # probability of churn
preds = model.predict(X_val)
accuracy = (preds == y_val).mean()

print(confusion_matrix(y_val, preds))
print(classification_report(y_val, preds))
```


### 📺 3.12 – Using the Final Model

**Objective:** Train on full train + val, predict on test

**Why retrain on train+val?**
To maximize the amount of data used before final testing. More data helps the model generalize better and improves performance on unseen examples.

```python
# 3.12 – Prepare full training set
df_full_train_dicts = df_full_train[categorical + numerical].to_dict(orient='records')
X_full_train = dv.fit_transform(df_full_train_dicts)
y_full_train = df_full_train.churn.values

# Final model training and scoring
final_model = LogisticRegression()
final_model.fit(X_full_train, y_full_train)

X_test = dv.transform(df_test[categorical + numerical].to_dict(orient='records'))
p_test = final_model.predict_proba(X_test)[:, 1]
y_pred = (p_test >= 0.5)
accuracy = (y_pred == y_test).mean()

print(classification_report(y_test, y_pred))
```

> Optional: Save model for deployment

```python
import joblib
joblib.dump((final_model, dv), 'model.joblib')
# Later load: final_model, dv = joblib.load('model.joblib')
```

---

### 📺 3.13 – Summary

**Covered Topics:**

* Data cleaning and EDA
* Churn rate, risk ratio, mutual information, and correlation
* One-hot encoding with `DictVectorizer`
* Logistic regression using `scikit-learn`
* Model evaluation and deployment-ready structure

> ✅ Accuracy: \~81% on test set
> 🔎 Most informative feature: `contract`, followed by `techsupport` and `internetservice`

---

**Interpretation of Feature Importance:**

* `contract` has the highest mutual information and a strong impact on churn: short-term or month-to-month contracts are more likely to churn.
* `techsupport` shows that customers without technical support are more likely to leave, making it a good behavioral predictor.
* `tenure` has a **negative correlation** with churn: longer-tenured customers are more stable.
* `monthlycharges` has a **positive correlation**, indicating that higher charges slightly increase churn risk.
* `gender`, by contrast, has **low mutual information** and **near-zero correlation**, making it statistically non-informative for predicting churn.

---

**Suggested Enhancements:**

* **Threshold Tuning**: Rather than using a 0.5 threshold, evaluate how varying it affects precision/recall. Helps optimize trade-offs.
* **Model Calibration**: Ensure predicted probabilities reflect actual churn likelihood using reliability plots or `calibration_curve`.
* **Class Imbalance**: Explore SMOTE or `class_weight='balanced'` to mitigate \~27% churn class imbalance.
* **Tree-Based Comparison**: Try `RandomForestClassifier` or `XGBoost` for better performance and feature interaction capture.
* **Pipeline Automation**: Use `Pipeline` + `ColumnTransformer` to cleanly chain preprocessing + modeling.
* **Business Perspective**: Map features like `contract` to actionable business steps (e.g., offer annual renewal to high-risk monthly users).
* **Cross-Validation**: Use `cross_val_score` or `StratifiedKFold` to validate model robustness across splits.

---

### 📊 Feature Importance Summary Table

| Feature        | Mutual Info | Correlation | Risk Ratio Notes           |
| -------------- | ----------- | ----------- | -------------------------- |
| contract       | High        | N/A         | Most predictive            |
| techsupport    | Medium      | N/A         | Absence → higher churn     |
| tenure         | N/A         | −0.35       | Long-term users churn less |
| monthlycharges | N/A         | +0.19       | Higher fees → more churn   |
| gender         | Low         | \~0         | Not predictive             |

---

## 📈 Pearson’s Correlation Coefficient

Used to check **linear relationships** between numerical features and the target.

While **mutual information** works well for **categorical** features, it’s not designed for **numerical** ones. For those, we turn to **Pearson’s correlation coefficient**, which tells us how strongly two numerical variables move together.

---

### 📐 What Is It?

Pearson's correlation coefficient (often noted as $\rho$ or just `corr`) measures **linear association** between two variables, $X$ and $Y$.

The result is a number between **−1 and 1**:

* **+1** → perfect positive linear relationship (as $X$ increases, $Y$ increases)
* **0** → no linear relationship
* **−1** → perfect negative linear relationship (as $X$ increases, $Y$ decreases)

**Formula:**

$$
\rho(X, Y) = \frac{\text{cov}(X, Y)}{\sigma_X \sigma_Y}
$$

Where:

* $\text{cov}(X, Y)$ = covariance of $X$ and $Y$
* $\sigma\_X$, $\sigma\_Y$ = standard deviations of $X$ and $Y$

---

### 🔍 What’s Covariance?

Covariance shows how much two variables vary together:

$$
\text{cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$

* If $X$ and $Y$ are usually both above or both below their means → **positive covariance**
* If one is high when the other is low → **negative covariance**


### 🧾 Standard Deviation Refresher

Standard deviation is how spread out values are around the mean:

$$
\sigma_X = \sqrt{\mathbb{E}[X^2] - \mathbb{E}[X]^2}, \quad \sigma_Y = \sqrt{\mathbb{E}[Y^2] - \mathbb{E}[Y]^2}
$$

So, dividing the covariance by the product of standard deviations **normalizes** the value between $-1$ and $1$.

---

### 🧪 Why Use Pearson’s Correlation?

* Helps detect linear patterns between numerical features and the target.
* Useful for **feature selection**: strong correlation might mean redundancy or importance.
* Can inform **scaling or transformation** choices (e.g., if relationship is nonlinear, try log/exp).

---

### 🧑‍💻 In Practice

In Section **3.7**, we used this to inspect how the numerical features (`tenure`, `monthlycharges`, `totalcharges`) correlate with `churn`.

```python
# 3.7 – Correlation matrix
correlation = df_full_train[numerical + ['churn']].corr()
print(correlation['churn'])
```

> 📊 Interpretation:
>
> * `tenure` is **negatively correlated** with churn → longer customers stay, the less likely they leave.
> * `monthlycharges` is **positively correlated** → higher charges may push customers to leave.
> * `totalcharges` may be misleading due to correlation with both tenure and charges.

---

### 📌 Where This Belongs

This content belongs right **after Section 3.6 (Mutual Information)** as Section **3.7** — which you already correctly placed in the summary.

---

## 🔍 Context: What Problem Are We Solving?

We are trying to **predict a binary outcome** (like "will a customer churn?" or "is this email spam?") based on input features. This is called a **binary classification** problem.

To solve it, we use **logistic regression**, which models the **probability** that the output is $1$ (i.e., the event happens), given some input features.

---

## 🧱 Dataset Setup

You start with a dataset structured like this:

| Instance $X\_0$ $X\_1$ ... $X\_d$ $Y$ |
| --------------------------------------------- |
| $x\_1$ $x\_{11}$ ... $x\_{1d}$ $y\_1$ |
| ...                                           |
| $x\_n$ $x\_{n1}$ ... $x\_{nd}$ $y\_n$ |

* $X\_0 = 1$ is the bias term.
* $\mathbf{X}$ is the **feature matrix**, including bias.
* $\mathbf{Y}$ is the **target vector**, taking values in ${0,1}$.

This is the **input** to our model. We want to model:

$$
p(Y=1|\mathbf{x})
$$

---

## 🚫 Why Not Use Linear Regression?

If we try:

$$
\pi(\mathbf{x}_i) = \mathbf{x}_i \cdot \mathbf{w}
$$

we're in trouble because the result is **not bounded** between 0 and 1. But a probability **must be**.

---

## ✅ Enter the Sigmoid

We fix this using the **sigmoid function**:

$$
\theta(z) = \frac{1}{1 + e^{-z}} \in [0, 1]
$$

Now we define:

$$
p(Y = 1|\mathbf{x}_i) = \theta(\mathbf{x}_i \cdot \mathbf{w})
$$

This gives a value between 0 and 1, perfect for a **probability**.

---

## ♻️ Properties of the Sigmoid

A useful identity:

$$
1 - \theta(z) = \theta(-z)
$$

So:

$$
p(Y = 1 | \mathbf{x}_i) = \theta(\mathbf{x}_i \cdot \mathbf{w})\\
p(Y = 0 | \mathbf{x}_i) = \theta(-\mathbf{x}_i \cdot \mathbf{w})
$$

General formula for both outcomes:

$$
p(Y|\mathbf{x}_i) = \theta(\mathbf{x}_i \cdot \mathbf{w})^{y_i} \cdot \theta(-\mathbf{x}_i \cdot \mathbf{w})^{1 - y_i}
$$

This is the **Bernoulli likelihood** of one data point.

---

## 🎯 Maximum Likelihood Estimation (MLE)

We now ask: what weights $\mathbf{w}$ make our observed data most likely?

### Likelihood:

$$
\mathcal{L}(\mathbf{w}) = \prod_{i=1}^{n} p(y_i | \mathbf{x}_i; \mathbf{w}) = \prod_{i=1}^{n} \theta(\mathbf{x}_i \cdot \mathbf{w})^{y_i} \cdot \theta(-\mathbf{x}_i \cdot \mathbf{w})^{1 - y_i}
$$

### Log-likelihood:

$$
\log \mathcal{L}(\mathbf{w}) = \sum_{i=1}^{n} \left[ y_i \log \theta(\mathbf{x}_i \cdot \mathbf{w}) + (1 - y_i) \log \theta(-\mathbf{x}_i \cdot \mathbf{w}) \right]
$$

Plugging in sigmoid:

$$
= \sum_{i=1}^{n} \left[ y_i \log \left( \frac{e^{\mathbf{x}_i \cdot \mathbf{w}}}{1 + e^{\mathbf{x}_i \cdot \mathbf{w}}} \right) + (1 - y_i) \log \left( \frac{1}{1 + e^{\mathbf{x}_i \cdot \mathbf{w}}} \right) \right]
$$

---

## ❌ Loss Function (to minimize)

We negate the log-likelihood:

$$
\text{loss}(\mathbf{w}) = -\log \mathcal{L}(\mathbf{w})
$$

This is the **binary cross-entropy**:

$$
\text{loss}(\mathbf{w}) = - \sum_{i=1}^{n} \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right]
$$

where $\hat{y}\_i = \theta(\mathbf{x}\_i \cdot \mathbf{w})$

---

## 💪 How Do We Minimize the Loss?

No closed-form solution. Use **iterative optimization**:

* Gradient descent (minimize loss)
* Or gradient ascent (maximize likelihood)

---

## ✅ Summary Flow

1. Prepare $\mathbf{X}$ and $\mathbf{Y}$ (add bias)
2. Model:

$$
p(Y = 1|\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{x} \cdot \mathbf{w}}}
$$

3. Likelihood:

$$
\prod_{i=1}^{n} \theta(\mathbf{x}_i \cdot \mathbf{w})^{y_i} \theta(-\mathbf{x}_i \cdot \mathbf{w})^{1 - y_i}
$$

4. Loss:

$$
\text{loss}(\mathbf{w}) = - \sum_{i=1}^{n} \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right]
$$

5. Optimize with gradient descent
6. Predict:

$$
\hat{y} = \begin{cases} 1 & \text{if } \theta(\mathbf{x} \cdot \mathbf{w}) > 0.5 \\ 0 & \text{otherwise} \end{cases}
$$

---

## 🧠 Understanding the Role of the Bias

If all features are zero:

$$
P(Y = 1 | \mathbf{x} = (1, 0, 0, 0)) = \theta(w_0) = \frac{e^{w_0}}{1 + e^{w_0}}
$$

This gives the **baseline probability** of the positive class.

Example:

$$
\theta(-0.577) \approx 0.36
$$

This means:

> Without knowing anything, the model predicts a 36% chance of churn.


## 🔍 Additional Work – Model Robustness, Scaling, and Regularization
### 🧪 Extra 1 – **Feature Scaling (Standardization)**
#### ✅ Motivation:

Numerical variables like `tenure` and `monthlycharges` are on **very different scales**, which can:

* Slow down convergence
* Bias the model if not scaled
* Trigger convergence warnings (e.g., for `lbfgs` solver)

#### 🛠️ How:

You used `StandardScaler` on numerical features to normalize them (zero mean, unit variance).

```python
scaler = StandardScaler()
X_train_num = scaler.fit_transform(df_train_full[numerical].values)
```

#### ✅ Impact:

* Helps optimization
* Improves accuracy
* Leads to better-conditioned models

---

### 🧪 Extra 2 – **Full Encoding Pipeline with OneHotEncoder + Scaler**

You replaced `DictVectorizer` with a more explicit and flexible pipeline:

* Categorical → `OneHotEncoder`
* Numerical → `StandardScaler`
* Combined → `np.column_stack([...])`

This is **clearer** and **production-friendly**, letting you easily plug into `Pipeline` or `GridSearchCV`.

#### Why:

* `DictVectorizer` mixes encoding and scaling
* `OneHotEncoder` gives more control (e.g., `handle_unknown='ignore'`)

```python
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat = ohe.fit_transform(df_train_full[categorical].values)
```

---

### 🧪 Extra 3 – **Regularization Strength (C) Tuning & Visualization**

You evaluated how the regularization parameter `C` affects:

* **Accuracy**
* **L2 norm of weights**

This helps you find the **sweet spot** between underfitting and overfitting.

#### 🔁 How:

You looped over `C = 10^{-10} \rightarrow 10^{20}`, plotting:

* Accuracy vs. `C`
* Weight norm vs. `C`

#### 📈 Results:

* Accuracy rises quickly for small `C` and plateaus around `C ≈ 0.1 → 1`
* Weight norm increases with `C` (as regularization weakens)

#### 🧠 Insight:

* Choose `C` in `[0.1, 1]` for best balance
* Large `C` yields unnecessary model complexity without accuracy gain

---

### 🧮 Extra 4 – **Condition Number Check**

You evaluated:

```python
np.linalg.cond(X_full_train.T @ X_full_train)
```

High condition number → poorly conditioned data (multicollinearity, instability in optimization).

✅ Despite poor conditioning, regularization helped maintain a **stable solution**.

---

### 🔍 Bonus – Logistic vs. Linear Output Visualization

You clearly illustrated why logistic regression uses **sigmoid**:

* Linear regression: outputs unbounded
* Logistic regression: uses sigmoid to output probabilities in $\[0, 1]\$

#### 📊 Graph:

* Shows sigmoid compressing extreme values
* Helps classify (e.g., using 0.5 threshold)

---

## 🧠 Takeaway Summary

| Area                     | What You Added                                        | Why It Matters                                            |
| ------------------------ | ----------------------------------------------------- | --------------------------------------------------------- |
| **Scaling**              | `StandardScaler` for numeric features                 | Improves training stability, convergence, and performance |
| **Encoding**             | Switched to `OneHotEncoder` for better pipeline logic | Better handling of unseen values, clearer split of duties |
| **Regularization Study** | Plotted Accuracy and Norm vs. `C`                     | Understand model complexity control and pick optimal `C`  |
| **Stability Check**      | Condition number of \$X^TX\$                          | Diagnoses ill-conditioning or multicollinearity           |
| **Theory Visualization** | Sigmoid vs Linear output curve                        | Shows logistic regression's probabilistic behavior        |


