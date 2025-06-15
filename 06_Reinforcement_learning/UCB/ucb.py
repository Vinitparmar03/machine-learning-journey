# ----------------------------
# Upper Confidence Bound (UCB)
# ----------------------------

# 📦 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# 📁 Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# 🚀 Implementing UCB Algorithm
N = 10000  # Total number of rounds (users)
d = 10     # Number of ads
ads_selected = []                         # To store ad selected in each round
numbers_of_selections = [0] * d           # Number of times each ad was selected
sums_of_rewards = [0] * d                 # Sum of rewards for each ad
total_reward = 0                          # Total reward accumulated
cumulative_rewards = []                   # To track cumulative reward over time

# 🔁 Loop over each round
for n in range(0, N):
    ad = 0
    max_ucb = 0
    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400  # A very large number to ensure each ad is selected at least once
        if upper_bound > max_ucb:
            max_ucb = upper_bound
            ad = i

    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
    cumulative_rewards.append(total_reward)

# 📊 Visualising the cumulative rewards over time
plt.figure(figsize=(12, 4))
plt.plot(range(N), cumulative_rewards)
plt.title('Cumulative Rewards Over Time')
plt.xlabel('Rounds')
plt.ylabel('Total Cumulative Reward')
plt.grid(True)
plt.tight_layout()
plt.show()

# 📊 Visualising the histogram of ads selections
plt.figure(figsize=(8, 5))
plt.hist(ads_selected, bins=np.arange(d + 1) - 0.5, edgecolor='black')
plt.title('Histogram of Ads Selections')
plt.xlabel('Ad Index')
plt.ylabel('Number of Times Each Ad Was Selected')
plt.xticks(range(d))
plt.grid(axis='y')
plt.tight_layout()
plt.show()
