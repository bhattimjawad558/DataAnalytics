# TASK 1: Overview & Summary Statistics
# Dataset: Netflix Movies and TV Shows
#  Import Required Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#  Load the Dataset
netflix = pd.read_csv('netflix_dataset.csv')
#  Preview the first few rows
print("Preview of Dataset:")
print(netflix.head(), "\n")
#  Dataset Information
print("Dataset Info:")
print(netflix.info(), "\n")
#  Summary Statistics (numerical + categorical)
print("Summary Statistics:")
print(netflix.describe(include='all'), "\n")
#  Missing Values
print("Missing Values per Column:")
print(netflix.isnull().sum(), "\n")
#  Distribution of Ratings
plt.figure(figsize=(10,6))
sns.countplot(y='rating', data=netflix, order=netflix['rating'].value_counts().index, palette='viridis')
plt.title('Distribution of Content Ratings on Netflix', fontsize=14)
plt.xlabel('Count')
plt.ylabel('Rating')
plt.show()
#  Distribution of Release Years
plt.figure(figsize=(12,6))
sns.histplot(netflix['release_year'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Netflix Titles by Release Year', fontsize=14)
plt.xlabel('Release Year')
plt.ylabel('Number of Titles')
plt.show()
#  Top 10 Most Common Genres
plt.figure(figsize=(10,6))
netflix['listed_in'].str.split(',').explode().str.strip().value_counts().head(10).plot(kind='barh', color='coral')
plt.title('Top 10 Most Common Genres on Netflix', fontsize=14)
plt.xlabel('Number of Titles')
plt.ylabel('Genre')
plt.show()
#  Top Countries Producing Content
plt.figure(figsize=(10,6))
netflix['country'].dropna().str.split(',').explode().str.strip().value_counts().head(10).plot(kind='bar', color='mediumseagreen')
plt.title('Top 10 Countries Producing Netflix Content', fontsize=14)
plt.xlabel('Number of Titles')
plt.ylabel('Country')
plt.show()
#  Quick Insights
print("Quick Insights:")
print(f"Total Titles: {len(netflix)}")
print(f"Number of Movies: {len(netflix[netflix['type']=='Movie'])}")
print(f"Number of TV Shows: {len(netflix[netflix['type']=='TV Show'])}")
print(f"Unique Ratings: {netflix['rating'].nunique()}")
print(f"Top Genre: {netflix['listed_in'].str.split(',').explode().str.strip().value_counts().idxmax()}")
print(f"Oldest Release Year: {netflix['release_year'].min()}")
print(f"Most Recent Release Year: {netflix['release_year'].max()}")
