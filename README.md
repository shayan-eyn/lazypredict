# Lazy Predict - Fork

This is a fork of the **Lazy Predict** repository, which introduces the following enhancements and modifications:

## Enhancements

### 1. **CatBoost Support**
   - **CatBoostRegressor** and **CatBoostClassifier** models are now enabled.
   - You can now use these models alongside existing Scikit-learn, XGBoost, and LightGBM models for both classification and regression tasks.

### 2. **New Metrics for Regression**
   - In addition to the default metrics, **Mean Absolute Error (MAE)** and **Mean Squared Error (MSE)** have been added for better performance evaluation of regression models.

### 3. **Verbosity Control**
   - **Verbosity** is turned off for LightGBM and CatBoost models to reduce logging noise and improve output readability.

### 4. **Preprocessing Toggle**
   - A new boolean parameter `preprocess` has been added to both `LazyClassifier` and `LazyRegressor`. This allows users to toggle preprocessing (scaling, imputation) on or off.
   - **When `preprocess=True`** (default): data is scaled and missing values are imputed.
   - **When `preprocess=False`**: raw data is used directly in the model pipeline.

## Installation

Install the repository directly from GitHub using `pip`:

```bash
pip install git+https://github.com/shayan-eyn/lazypredict.git
```

## Usage

To use the new features, follow the same pattern as the original repository, with the added options for `preprocess`, CatBoost models, and additional regression metrics.

### Example for Classification

```python
from Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

clf = LazyClassifier(preprocess=True)  # Toggle preprocessing with the `preprocess` parameter
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)
```

### Example for Regression

```python
from Supervised import LazyRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

data = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

reg = LazyRegressor(preprocess=False)  # Preprocessing disabled
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)
```

## Contributions

Feel free to contribute to this fork by submitting a pull request with your enhancements or bug fixes.


