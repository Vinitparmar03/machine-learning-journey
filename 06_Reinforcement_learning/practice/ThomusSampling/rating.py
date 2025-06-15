import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from collections import Counter
import random

# Load datasets
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge ratings with movie titles
merged = ratings.merge(movies, on='movieId')

# Define binary interaction: 1 if rating >= 3, else 0
merged['interaction'] = (merged['rating'] >= 3).astype(int)

# Select top 10 most rated movies
top_movies = merged['title'].value_counts().head(10).index.tolist()
filtered = merged[merged['title'].isin(top_movies)]

# Keep only users who have rated at least 3 of these top movies
user_counts = filtered.groupby('userId')['title'].nunique()
active_users = user_counts[user_counts >= 3].index
filtered = filtered[filtered['userId'].isin(active_users)]

# Create dense user-movie interaction matrix
pivot_df = filtered.pivot_table(index='userId', columns='title', values='interaction', fill_value=0)

# Basic info
columns = pivot_df.columns.tolist()
N = pivot_df.shape[0]
d = pivot_df.shape[1]

# Initialize Thompson Sampling parameters
movies_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0

# Thompson Sampling algorithm
for n in range(N):
    movie = 0
    max_random = 0
    for i in range(d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            movie = i

    rwd = pivot_df.iloc[n, movie]
    movies_selected.append(columns[movie])
    if rwd == 1:
        numbers_of_rewards_1[movie] += 1
    else:
        numbers_of_rewards_0[movie] += 1
    total_reward += rwd

# Summary of results
print("Total Reward:", total_reward)
selection_counts = Counter(movies_selected)
selection_df = pd.DataFrame(selection_counts.items(), columns=['Movie Title', 'Selections']).sort_values(by='Selections', ascending=False)

print("\nTop Selected Movies:")
print(selection_df)

# Visualization 1: Number of times each movie was selected
plt.figure(figsize=(10, 5))
plt.bar(columns, [selection_counts.get(title, 0) for title in columns], color='skyblue')
plt.title('Number of Selections per Movie (Thompson Sampling)')
plt.xlabel('Movie Title')
plt.ylabel('Number of Selections')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Visualization 2: Histogram of selected movies
plt.figure(figsize=(10, 5))
plt.hist(movies_selected, bins=np.arange(len(columns) + 1) - 0.5, rwidth=0.8, color='lightgreen')
plt.title('Histogram of Selected Movies')
plt.xlabel('Movie Title')
plt.ylabel('Frequency')
plt.xticks(ticks=range(len(columns)), labels=columns, rotation=90)
plt.tight_layout()
plt.show()
