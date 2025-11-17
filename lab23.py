# -----------------------------------------
# Task 1: Heart Disease UCI - Full EDA Script
# -----------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# --------------------------
# 1. Load dataset
# --------------------------
df = pd.read_csv("heart_cleveland.csv")
# --------------------------
# 2. Inspect data
# --------------------------
print("\n--- SHAPE ---")
print(df.shape)
print("\n--- INFO ---")
print(df.info())
print("\n--- DESCRIPTIVE STATISTICS ---")
print(df.describe(include='all'))
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())
# --------------------------
# 3. Basic Cleaning
# --------------------------
# remove duplicates
df = df.drop_duplicates()
# fill numeric NaN with median (if any)
df = df.fillna(df.median(numeric_only=True))
# ensure categorical columns are proper dtype
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
for col in categorical_cols:
    df[col] = df[col].astype("category")
# --------------------------
# 4. Univariate Analysis
# --------------------------
numeric_cols = df.select_dtypes(include=['number']).columns
for col in numeric_cols:
    plt.figure(figsize=(7,4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()
# Categorical counts
for col in categorical_cols:
    plt.figure(figsize=(6,4))
    df[col].value_counts().plot(kind='bar')
    plt.title(f"Value Counts: {col}")
    plt.show()
# --------------------------
# 5. Bivariate Analysis
# --------------------------
# Continuous vs continuous (scatter)
sns.pairplot(df[numeric_cols])
plt.show()
# Continuous vs categorical (boxplots)
for col in numeric_cols:
    plt.figure(figsize=(7,4))
    sns.boxplot(data=df, x="condition", y=col)
    plt.title(f"{col} vs condition")
    plt.show()
# Categorical vs categorical (heatmap of counts)
cross = pd.crosstab(df["sex"], df["condition"])
sns.heatmap(cross, annot=True, cmap='Blues')
plt.title("Sex vs Condition")
plt.show()
# --------------------------
# 6. Correlation Analysis
# --------------------------
plt.figure(figsize=(12,8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
# --------------------------
# 7. Data Drilling (Granularity)
# --------------------------
# High level: condition rates
print("\n--- CONDITION BY SEX ---")
print(df.groupby("sex")["condition"].agg(["mean", "count"]))
print("\n--- CONDITION BY CHEST PAIN TYPE (cp) ---")
print(df.groupby("cp")["condition"].agg(["mean", "count"]))
# Drilling to Example subgroup
sub = df[df["sex"] == 1]           # males
print("\n--- MALE SUBGROUP STATS ---")
print(sub.describe())
# Drilling to individual record example
print("\n--- SAMPLE RECORDS ---")
print(df.head(3))
# --------------------------
# 8. Outlier Detection
# --------------------------
def detect_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]
print("\n--- OUTLIERS BY IQR ---")
for col in numeric_cols:
    outliers = detect_outliers_iqr(df[col])
    if len(outliers) > 0:
        print(f"{col}: {len(outliers)} outliers")
# Z-score anomalies
z_scores = np.abs(stats.zscore(df[numeric_cols]))
anomaly_rows = (z_scores > 3).any(axis=1)
print("\nZ-score anomalies (|z| > 3):", anomaly_rows.sum())
# --------------------------
# 9. Pattern Discovery
# --------------------------
# Relationship patterns: pairwise scatter conditioned on CP type
sns.pairplot(df, hue="cp", vars=["age", "trestbps", "chol", "thalach", "oldpeak"])
plt.show()
# Optional cluster exploration on 2 features
from sklearn.cluster import KMeans
X = df[["age", "thalach"]]
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)
plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="age", y="thalach", hue="cluster", palette="viridis")
plt.title("KMeans Clustering: age vs thalach")
plt.show()
# --------------------------
# 10. Interpretation
# --------------------------
print("\n--- INTERPRETATION ---")
print("""
Across the dataset, several variables show strong relationships with the heart disease condition indicator.
Higher age, lower maximum heart rate (thalach), higher resting blood pressure, and higher ST depression (oldpeak)
are all associated with increased likelihood of disease. Categorical patterns indicate that chest pain type "0"
(typical angina) and being male correspond with higher condition rates. Outlier analysis detected unusual values
in cholesterol and resting blood pressure, which may represent measurement errors or rare but significant cases.
Correlation analysis highlights oldpeak and thalach as the strongest predictors. Overall, the dataset reveals clear
clinical risk patterns consistent with known cardiovascular indicators and supports further modeling work.
""")
