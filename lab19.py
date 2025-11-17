# Task 5: Netflix Dataset Analysis & Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 1. Load dataset
df = pd.read_csv("netflix_dataset.csv")
# 2. Convert 'date_added' to datetime
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
# 3. Inspect dataset
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
# 4. Basic statistics for numeric columns
print("\nDescriptive Statistics:")
print(df.describe())
# 5. Unique values in categorical columns
print("\nUnique Types:", df['type'].unique())
print("\nUnique Ratings:", df['rating'].unique())
print("\nUnique Countries (sample):", df['country'].dropna().unique()[:10])
# 6. Count of TV Shows vs Movies
plt.figure(figsize=(6,4))
sns.countplot(x='type', data=df, palette='Set2')
plt.title("Count of TV Shows vs Movies")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()
# 7. Top 10 countries with most content
top_countries = df['country'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='viridis')
plt.title("Top 10 Countries by Number of Shows/Movies")
plt.xlabel("Number of Shows/Movies")
plt.ylabel("Country")
plt.show()
# 8. Distribution of release years
plt.figure(figsize=(12,5))
sns.histplot(df['release_year'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Release Years")
plt.xlabel("Release Year")
plt.ylabel("Count")
plt.show()
# 9. Shows added per year
df['year_added'] = df['date_added'].dt.year
plt.figure(figsize=(12,5))
df['year_added'].value_counts().sort_index().plot(kind='bar', color='coral')
plt.title("Number of Shows/Movies Added Per Year")
plt.xlabel("Year Added")
plt.ylabel("Count")
plt.show()
# 10. Most common genres
# 'listed_in' can have multiple genres separated by commas
from collections import Counter
genre_list = df['listed_in'].dropna().apply(lambda x: [g.strip() for g in x.split(',')])
flat_genres = [genre for sublist in genre_list for genre in sublist]
top_genres = Counter(flat_genres).most_common(10)
genres, counts = zip(*top_genres)
plt.figure(figsize=(10,5))
sns.barplot(x=list(counts), y=list(genres), palette='magma')
plt.title("Top 10 Most Common Genres")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.show()
# 11. Insights & Interpretation
print("""
INSIGHTS & INTERPRETATION:
1. The dataset contains both TV Shows and Movies, with varying distributions.
2. Most content comes from the US, followed by countries like India, UK, and Japan.
3. Release years show content spanning many decades; peaks may indicate popular production years.
4. Number of shows/movies added per year shows Netflix's growth over time.
5. Top genres include International TV Shows, Dramas, Comedies, and Documentaries, indicating user preference trends.
6. 'date_added' and 'release_year' help differentiate between production and platform addition, useful for trend analysis.
7. Business Implication: Understanding popular genres, regions, and content types can guide content acquisition and production strategies.
""")
