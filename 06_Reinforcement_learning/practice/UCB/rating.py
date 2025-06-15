import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from collections import Counter

movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

merged = ratings.merge(movies, on='movieId')
merged['interaction'] = (merged['rating'] >= 3).astype(int)

# Select top 10 most rated movies
top_movies = merged['title'].value_counts().head(10).index.tolist()
filtered = merged[merged['title'].isin(top_movies)]

# Keep only users who have rated at least 3 of these movies
user_counts = filtered.groupby('userId')['title'].nunique()
active_users = user_counts[user_counts >= 3].index
filtered = filtered[filtered['userId'].isin(active_users)]

# Pivot table (dense subset)
pivot_df = filtered.pivot_table(index='userId', columns='title', values='interaction', fill_value=0)

columns = pivot_df.columns.tolist()

N = pivot_df.shape[0]
d = pivot_df.shape[1]

movies_selected = []
no_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0

for n in range(N):
    movie = 0
    max_ucb = 0
    for i in range(d):
        if no_of_selections[i] > 0:
            avg_reward = sum_of_rewards[i] / no_of_selections[i]
            delta_i = math.sqrt(3 / 2 * math.log(n + 1) / no_of_selections[i])
            ucb = avg_reward + delta_i
        else:
            ucb = 999999
        if ucb > max_ucb:
            max_ucb = ucb
            movie = i

    rwd = pivot_df.iloc[n, movie]
    movies_selected.append(columns[movie])
    no_of_selections[movie] += 1
    sum_of_rewards[movie] += rwd
    total_reward += rwd

# Summary
print("Total Reward:", total_reward)
selection_counts = Counter(movies_selected)
selection_df = pd.DataFrame(selection_counts.items(), columns=['Movie Title', 'Selections']).sort_values(by='Selections', ascending=False)

print("\nTop Selected Movies:")
print(selection_df)

# Visualization 1: Number of times each movie was selected
plt.figure(figsize=(10, 5))
plt.bar(columns, no_of_selections, color='skyblue')
plt.title('Number of Selections per Movie (UCB)')
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