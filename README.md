# 🧠 Crime Rate Prediction using Genetic Algorithm and Regression Models

This project implements a machine learning pipeline that predicts crime rates using selected features optimized via a **Genetic Algorithm (GA)**. Multiple regression models are evaluated to compare performance, and the best feature set is used for final prediction.

---

## 📊 Objective

To predict crime rates efficiently by:
- Selecting the most relevant crime-related features using Genetic Algorithms.
- Applying regression models like Ridge, Lasso, Random Forest, Gradient Boosting, and XGBoost.
- Comparing performance and interpreting model results.

---

## 🔧 Technologies Used

- Python 3.x
- Scikit-learn
- XGBoost
- DEAP (Distributed Evolutionary Algorithms in Python)
- NumPy / Pandas / Matplotlib

---

## 🧬 Feature Selection (Genetic Algorithm)

The GA optimized the feature set for model performance using the following metrics:
- **R^2 Score**
- **Mean Absolute Error (MAE)**

### ✅ Selected Features:
```python
['RAPE', 'DACOITY', 'PREPARATION AND ASSEMBLY FOR DACOITY', 'RIOTS', 'CHEATING', 'INSULT TO MODESTY OF WOMEN']
