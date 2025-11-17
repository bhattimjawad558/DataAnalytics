# -----------------------------------------------------------
# TASK 2 – Hypothesis Testing + Aggregation + Drilldown
# World Energy Dataset (Full Columns Provided by User)
# -----------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv("World Energy Consumption.csv")
print("\n===== DATA PREVIEW =====")
print(df.head())
print(df.info())
# ==========================================
# 2. BASIC CLEANING
# ==========================================
# Convert year to numeric
df['year'] = pd.to_numeric(df['year'], errors='coerce')
# Replace negative / impossible values with NaN
df['energy_per_capita'] = df['energy_per_capita'].replace([-np.inf, np.inf], np.nan)
# Drop rows with missing country names or energy values
df_clean = df.dropna(subset=['country', 'energy_per_capita'])
# ==========================================
# 3. CLASSIFY COUNTRIES – Developed vs Developing
# Using UN Human Development Index / known country lists
# ==========================================
developed_list = [
    "United States", "Canada", "United Kingdom", "Germany", "France", "Japan",
    "South Korea", "Australia", "New Zealand", "Sweden", "Norway", "Finland",
    "Denmark", "Netherlands", "Belgium", "Switzerland", "Austria", "Singapore"
]
df_clean['status'] = df_clean['country'].apply(
    lambda x: "Developed" if x in developed_list else "Developing"
)
# ==========================================
# 4. HYPOTHESIS TESTING
# H0: Developed countries consume same energy as developing
# H1: Developed countries consume MORE energy
# ==========================================
dev = df_clean[df_clean['status']=="Developed"]['energy_per_capita'].dropna()
deving = df_clean[df_clean['status']=="Developing"]['energy_per_capita'].dropna()
t_stat, p_val = ttest_ind(dev, deving, equal_var=False)
print("\n===== HYPOTHESIS TEST RESULTS =====")
print("T-statistic:", t_stat)
print("P-value:", p_val)
if p_val < 0.05:
    print("➡ RESULT: Reject H0 — Developed countries consume significantly MORE energy.")
else:
    print("➡ RESULT: Fail to reject H0 — No significant difference detected.")
# ==========================================
# 5. AGGREGATION ANALYSIS
# ==========================================
print("\n===== TOTAL ENERGY BY COUNTRY =====")
print(df_clean.groupby("country")['primary_energy_consumption'].sum().sort_values(ascending=False).head(10))
print("\n===== MEAN ENERGY PER CAPITA BY COUNTRY =====")
print(df_clean.groupby("country")['energy_per_capita'].mean().sort_values(ascending=False).head(10))
print("\n===== MAX ENERGY GENERATION BY COUNTRY =====")
print(df_clean.groupby("country")['electricity_generation'].max().sort_values(ascending=False).head(10))
# ==========================================
# 6. OUTLIER DETECTION (Z-score)
# ==========================================
df_clean['z_energy'] = (df_clean['energy_per_capita'] - df_clean['energy_per_capita'].mean()) / df_clean['energy_per_capita'].std()
outliers = df_clean[df_clean['z_energy'] > 3]
print("\n===== OUTLIER COUNTRIES (Very High Energy Use) =====")
print(outliers[['country','year','energy_per_capita']].head())
# ==========================================
# 7. DRILL-DOWN ANALYSIS (Country → Year)
# ==========================================
print("\n===== DRILL DOWN: ENERGY TREND FOR USA =====")
usa = df_clean[df_clean['country']=="United States"].sort_values("year")
print(usa[['year','energy_per_capita','primary_energy_consumption']].head(15))
# ==========================================
# 8. CORRELATION ANALYSIS
# ==========================================
corr_cols = ['energy_per_capita','gdp','primary_energy_consumption',
             'electricity_generation','population']
corr_df = df_clean[corr_cols]
corr_matrix = corr_df.corr()
print("\n===== CORRELATION MATRIX =====")
print(corr_matrix)
# ==========================================
# 9. CORRELATION HEATMAP GRAPH
# ==========================================
plt.figure(figsize=(10,7))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("World Energy Dataset – Correlation Heatmap")
plt.show()
# ==========================================
# 10. SCATTER PLOTS
# ==========================================
# GDP vs Energy per Capita
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_clean, x='gdp', y='energy_per_capita', hue='status', alpha=0.5)
plt.title("GDP vs Energy Per Capita")
plt.xlabel("GDP")
plt.ylabel("Energy Per Capita")
plt.show()
# Primary Energy Consumption vs Electricity Generation
plt.figure(figsize=(8,6))
sns.scatterplot(data=df_clean, x='primary_energy_consumption', y='electricity_generation', alpha=0.5)
plt.title("Energy Consumption vs Electricity Generation")
plt.xlabel("Primary Energy Consumption")
plt.ylabel("Electricity Generation")
plt.show()
