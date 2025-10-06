# 🎥 ML Zoomcamp 2.X - Car Price Prediction - Comprehensive Lecture Notes

## 📄 Project Context

The main goal of this project was to predict car prices using a machine learning model built from scratch. The data used comes from a Kaggle dataset, and we walk through all the steps typically required in a regression project: data cleaning, preprocessing, model training, evaluation, regularization, and deployment logic.

---

## 📘 Session 2 – Hybrid Walkthrough & Deep Dive

### 📌 1. Dataset Overview and Normalization

We begin by loading a real-world car dataset, which contains 16 columns describing different attributes (brand, model, horsepower, MPG, etc.) and the **target variable**: `msrp` (price).

#### 🧪 Code

```python
df = pd.read_csv('Data/data_car_price.csv')
```

#### 🔍 Problem Noted:

* Inconsistent formatting in column names (spaces, capitalization)
* Inconsistent categorical formatting in rows (e.g., "Premium Unleaded" vs "premium unleaded")

#### ✅ Normalization Steps:

* Column names converted to lowercase and spaces replaced by underscores
* String columns cleaned using `.str.lower()` and `.str.replace(' ', '_')`

#### 📌 Key Takeaways:

* Clean column names = easier to work with programmatically
* Normalized categorical values = consistency for feature engineering

---

### 📌 2. Target Distribution Analysis

Understanding `msrp` is crucial. The initial distribution is **highly skewed** due to a few extremely expensive cars.

#### 🧪 Visual:

```python
sns.histplot(df.msrp[df.msrp < 1e5])
```

#### 🔍 Insight:

* The **long-tail distribution** could distort regression
* Solution: apply `log1p()` transform to compress the range and normalize the spread

#### ✅ Why `np.log1p()`?

* Handles zero values safely
* Improves model stability

---

### 📌 3. Missing Values

We detect missing values in several columns, with `market_category` having **\~30% missingness**.

#### 🧪 Code:

```python
df.isnull().sum()
```

#### 🔍 Insight:

* Small missing values in `engine_hp`, `engine_cylinders`, `number_of_doors` can be filled
* Large-scale missingness (like in `market_category`) requires careful consideration (e.g., ignore or encode as “unknown”)

---

### 📌 4. Train/Val/Test Split

A **randomized 60/20/20 split** is created with shuffling to prevent order bias.

#### 🧪 Code Logic:

```python
np.random.seed(2)
# split and shuffle logic...
```

#### ✅ Tip:

Always keep raw `msrp` for back-transformation after modeling.

---

### 📌 5. Linear Regression: Baseline

We implement a **manual closed-form solution** for multiple linear regression using NumPy:

#### 🧪 Function:

```python
def linear_regression(X, y):
    ...
```

#### 📉 Initial performance:

* RMSE (Validation): \~0.76
* Only basic numeric features used (`engine_hp`, `city_mpg`, etc.)
* Missing values filled with zeros

#### 🔍 Insight:

The model underfits due to insufficient and shallow features


---

## 📐 Linear Regression: Matrix Formulation and Optimization

To understand the internal mechanics of linear regression, we can formulate it in vectorized (matrix) form for efficient computation and mathematical clarity.

### 🧾 Dataset Representation

Consider a dataset with \$n\$ samples and \$d\$ features. We augment each feature vector with an intercept term (bias), denoted as \$X\_0 = 1\$. The dataset can be represented as:

$$
\left( 
\begin{array}{c|cccc|c}
~    &X_0&X_1&\cdots & X_d  & Y \\
\hline
x_1 &1& x_{11}& \cdots&x_{1d}&y_1 \\
\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\
x_n&1&x_{n1}&\cdots&x_{nd}&y_n
\end{array} 
\right)
$$

We extract two components:

* **Feature matrix** \$\mathbf{X}\$:

$$
\mathbf{X}=
\left( 
\begin{array}{cccc}
1 & x_{11} & \cdots & x_{1d} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & \cdots & x_{nd}
\end{array} 
\right)
$$

* **Target vector** \$\mathbf{Y}\$:

$$
\mathbf{Y} = 
\left( 
\begin{array}{c}
y_1 \\
\vdots \\
y_n
\end{array} 
\right)
$$

---

### 🧠 Model Prediction

Linear regression models the relationship as:

$$
\hat{\mathbf{Y}} = \mathbf{X} \mathbf{w}
$$

Where:

* \$\mathbf{w} = (w\_0, \ldots, w\_d)^T\$ are the model parameters (weights)
* \$\hat{\mathbf{Y}}\$ are the predicted values

---

### 🧮 Loss Function: Sum of Squared Errors (SSE)

To learn the optimal weights, we minimize the error between prediction and ground truth using SSE:

$$
\text{SSE} = \sum_{i=1}^{n} \epsilon_i^2 = \mathbf{\epsilon}^T \mathbf{\epsilon}
$$

Where:

* \$\boldsymbol{\epsilon} = \hat{\mathbf{Y}} - \mathbf{Y} = \mathbf{Xw} - \mathbf{Y}\$

Expanding the squared error:

$$
\begin{aligned}
\text{SSE} 
&= (\mathbf{Xw} - \mathbf{Y})^T (\mathbf{Xw} - \mathbf{Y}) \\
&= \mathbf{Y}^T\mathbf{Y} - 2\mathbf{Y}^T\mathbf{Xw} + \mathbf{w}^T\mathbf{X}^T\mathbf{Xw}
\end{aligned}
$$

---                                                                                                                                                         

### 🧬 Optimization: Deriving the Normal Equation

To minimize SSE, take the derivative of the loss function w\.r.t. \$\mathbf{w}\$ and set it to zero:

$$
\frac{\partial \text{SSE}}{\partial \mathbf{w}} = -2\mathbf{X}^T\mathbf{Y} + 2\mathbf{X}^T\mathbf{Xw} = 0
$$

Solving for \$\mathbf{w}\$:

$$
\boxed{
\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}
}
$$

This is called the **normal equation**, and it gives us the closed-form solution for the optimal weight vector.

---

### 🔧 Python Implementation Sketch

```python
def linear_regression(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y
```

---


### 📌 6. Feature Engineering: Add Age

We add a new numeric feature `age = 2017 - year`.

#### ✅ Result:

* RMSE drops significantly to **\~0.51**
* Shows that **domain-informed features** can improve performance

---

### 📌 7. Categorical Feature Encoding (One-Hot Manual)
#### 📌 Traitement des variables catégorielles

Les **variables catégorielles** contiennent des informations qualitatives (ex: marque, carburant). Elles doivent être **converties en valeurs numériques** pour être utilisées dans les modèles de régression.

#### ➕ Méthode utilisée : **One-Hot Encoding**

* Chaque catégorie devient une **colonne binaire** (0 ou 1).
* Exemple : la colonne `carburant` contenant *essence*, *diesel*, *électrique* devient trois colonnes : `essence`, `diesel`, `électrique`.

#### ⚠️ Limitations :

* Trop de catégories = explosion du nombre de colonnes.
* Certaines catégories peuvent être **corrélées**, ce qui crée une **colinéarité** dans la matrice \$X\$.

  * Cela peut rendre la matrice \$(X^T X)\$ **non inversible**, affectant les performances du modèle.

#### ✅ Solution :

* Limiter les catégories à celles les plus fréquentes.
* Utiliser la **régularisation (Ridge)** pour stabiliser les calculs et réduire l’impact des variables corrélées.

We encode:

* `make` (top 5)
* `number_of_doors` (2, 3, 4)
* `engine_fuel_type` (top 4)
* Plus `transmission`, `driven_wheels`, `vehicle_size`, `vehicle_style`, etc.

#### 🔍 Insight:

Adding too many features without regularization **worsens RMSE** due to:

* Multicollinearity
* Numerical instability

---

### 📌 8. Regularization: Ridge Regression (L2)

To fix instability from correlated inputs, we apply **Ridge Regression**, which adds a penalty term to the weights.

#### 🧪 Function:

```python
def ridge_regression(X, y, alpha):
    ...
```

#### ✅ Benefit:

* Stabilizes the matrix inversion
* Controls the magnitude of weights
* Makes the model robust to overfitting

#### 📉 Results:

* Optimal alpha ≈ `0.01`
* Final RMSE (Validation): **\~0.46**
* Test RMSE: **\~0.45** — indicating strong generalization

Here is a clean, well-structured **Markdown version** of your explanation about **Ridge Regression** and **regularization**, formatted consistently with your lecture notes:

---

## 🧮 Ridge Regression: Handling Multicollinearity with Regularization

When adding new features to a linear regression model, performance can sometimes degrade rather than improve. This is due to **numerical instability** and **multicollinearity** — when features are highly correlated.

### ❌ The Problem: Invertibility of $\mathbf{X}^T \mathbf{X}$

The normal equation for linear regression:

$$
\mathbf{w} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}
$$

relies on the assumption that $\mathbf{X}^T \mathbf{X}$ is invertible. However, when features are **not orthogonal** (i.e., correlated), this matrix can become:

* **Ill-conditioned**: nearly singular, leading to large numerical errors
* **Singular**: non-invertible, making the equation unsolvable

This leads to **unstable** or **exaggerated weights**, harming model generalization.

---

### ✅ The Solution: Ridge Regression (L2 Regularization)

To ensure invertibility and stabilize weight estimation, we add a **penalty term** on the size of the weights:

### Regularized Loss Function:

$$
L(\mathbf{w}) = \| \mathbf{Y} - \hat{\mathbf{Y}} \|^2 + \alpha \| \mathbf{w} \|^2
$$

Where:

* $\alpha$ is the **regularization strength**
* $\| \mathbf{w} \|^2 = \sum w_i^2$ penalizes large weights

---

### 🔍 Optimizing the Regularized Loss

Taking the derivative with respect to $\mathbf{w}$ and setting it to zero:

$$
\frac{dL(\mathbf{w})}{d\mathbf{w}} = -2\mathbf{X}^T\mathbf{Y} + 2(\mathbf{X}^T\mathbf{X})\mathbf{w} + 2\alpha \mathbf{w} = 0
$$

Solving for $\mathbf{w}$:

$$
\boxed{
\mathbf{w} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{Y}
}
$$

This is known as the **Ridge Regression** solution.

---

###     🔒 Why Is This Always Invertible?

* $\mathbf{X}^T \mathbf{X}$: symmetric and positive semi-definite (eigenvalues ≥ 0)
* $\alpha \mathbf{I}$: diagonal matrix with **strictly positive** eigenvalues
* Therefore, $\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I}$: symmetric and **positive definite** (eigenvalues > 0)

➡️ A **positive definite matrix is always invertible**, guaranteeing stable weight estimation.

---

### 🧪 Python Implementation

```python
def ridge_regression(X, Y, alpha=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    I = np.eye(XTX.shape[0])
    XTX_inv = np.linalg.inv(XTX + alpha * I)

    w = XTX_inv.dot(X.T).dot(Y)
    
    return w[0], w[1:]
```


---

### 📌 9. Final Prediction Example

Model successfully predicts a price close to actual MSRP using a single row from `df_test`.

#### 🧪 Logic:

```python
np.expm1(y_pred)  # back from log scale
```

#### 🔍 Insight:

Even a simple linear model with well-engineered features + regularization can give **practically useful predictions**

---

## ✅ Summary: Key Gains from Session 2

* Learned to **prepare and normalize a real-world dataset**
* Understood **how skewed targets affect regression** and the benefit of `log1p`
* Built a **manual linear regression model from scratch**
* Gradually added features and saw the impact of engineering
* Applied **Ridge regression** to stabilize and improve the model
* Validated predictions using RMSE and price back-transformation

---

➡️ **Next step in ML Zoomcamp**: Dive into **classification problems** using **scikit-learn** and structured APIs.
