# ---------------------------------------------------------
# MULTIPLE LINEAR REGRESSION - MEDICAL INSURANCE COST
# Dataset: insurance.csv
# Predictors: age, bmi, smoker, region
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------
df = pd.read_csv("medical_insurance.csv")
print(df.head())
# ---------------------------------------------------------
# 2. ENCODE CATEGORICAL VARIABLES
# ---------------------------------------------------------
# Convert "smoker" to binary: yes=1, no=0
df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})
# One-hot encode "region" (drop_first=True avoids multicollinearity)
df = pd.get_dummies(df, columns=["region"], drop_first=True)
# ---------------------------------------------------------
# 3. SELECT FEATURES
# ---------------------------------------------------------
X = df[["age", "bmi", "smoker", 
        "region_northwest", "region_southeast", "region_southwest"]]
y = df["charges"]
# ---------------------------------------------------------
# 4. TRAIN-TEST SPLIT & FIT MODEL
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
# ---------------------------------------------------------
# 5. COEFFICIENTS
# ---------------------------------------------------------
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})
print("\n--- MODEL COEFFICIENTS ---")
print(coef_df)
print("\nIntercept (b0):", model.intercept_)
# ---------------------------------------------------------
# 6. MODEL EVALUATION
# ---------------------------------------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\n--- MODEL PERFORMANCE ---")
print("RMSE:", rmse)
print("R² Score:", r2)
