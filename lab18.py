# Task 1: Load dataset using pandas
import pandas as pd
# Load dataset (adjust path if needed)
df = pd.read_csv('Global_Superstore.csv', encoding='ISO-8859-1')
# Display basic info
print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nPreview of Dataset:")
print(df.head())
# Task 2: Display unique categorical values
print("Unique Categories:", df['Category'].unique())
print("Unique Regions:", df['Region'].unique())
print("Unique Segments:", df['Segment'].unique())
# Task 3: Filtering / Searching
# Example: Filter all 'Office Supplies' sold in 'West' region
filtered_data = df[(df['Category'] == 'Office Supplies') & (df['Region'] == 'West')]
print("Filtered Rows:", len(filtered_data))
print(filtered_data[['Category', 'Region', 'Sales', 'Profit', 'Customer.Name']].head())
# Task 4: Compute aggregation metrics
sales = df['Sales']
profit = df['Profit']
print("=== Sales Metrics ===")
print("Count:", sales.count())
print("Total Sales:", round(sales.sum(), 2))
print("Mean:", round(sales.mean(), 2))
print("Median:", round(sales.median(), 2))
print("Mode:", round(sales.mode()[0], 2))
print("Standard Deviation:", round(sales.std(), 2))
print("Min:", round(sales.min(), 2))
print("Max:", round(sales.max(), 2))
print("\n=== Profit Metrics ===")
print("Total Profit:", round(profit.sum(), 2))
print("Average Profit:", round(profit.mean(), 2))
print("Standard Deviation:", round(profit.std(), 2))
# Aggregation for the subset
agg_results = {
    'Count': subset['Sales'].count(),
    'Total Sales': subset['Sales'].sum(),
    'Average Sales': subset['Sales'].mean(),
    'Median Sales': subset['Sales'].median(),
    'Mode Sales': subset['Sales'].mode()[0],
    'Std Dev (Sales)': subset['Sales'].std(),
    'Min Sales': subset['Sales'].min(),
    'Max Sales': subset['Sales'].max(),
    'Total Profit': subset['Profit'].sum(),
    'Average Profit': subset['Profit'].mean(),
    'Std Dev (Profit)': subset['Profit'].std()
}
for k, v in agg_results.items():
    print(f"{k}: {v}")

