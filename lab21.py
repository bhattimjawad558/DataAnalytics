# --- Task 1: Overview & Summary Statistics (Netflix Movies Dataset) ---
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load dataset
netflix = pd.read_csv('netflix_dataset.csv')
# Display summary statistics (includes object columns)
print("===== SUMMARY STATISTICS =====")
print(netflix.describe(include='all'))
# Countplot for Ratings
plt.figure(figsize=(10,6))
sns.countplot(y='rating', data=netflix, order=netflix['rating'].value_counts().index)
plt.title('Distribution of Content Ratings')
plt.xlabel('Count')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()
