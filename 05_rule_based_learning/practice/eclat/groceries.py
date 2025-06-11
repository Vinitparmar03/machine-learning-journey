# ---------------------- Import Required Libraries ----------------------

import numpy as np                    # For numerical operations like mean, min, max
import pandas as pd                   # For data manipulation and analysis
from apyori import apriori            # To apply the Apriori algorithm for association rule mining

# ---------------------- Define Function to Extract Rules ----------------------

def inspect(results):
    # Extract left-hand side, right-hand side, and support of rules
    lhs = [tuple(r.ordered_statistics[0].items_base)[0] for r in results]
    rhs = [tuple(r.ordered_statistics[0].items_add)[0] for r in results]
    support = [r.support for r in results]
    return list(zip(lhs, rhs, support))

# ---------------------- Load and Prepare Dataset ----------------------

df = pd.read_csv('Groceries_dataset.csv')                     # Load the dataset

# Create a unique transaction ID by combining member number and date
df['Transaction'] = df['Member_number'].astype(str) + '_' + df['Date']

# Group items by transaction to get list of items bought together
grouped = df.groupby('Transaction')['itemDescription'].apply(list)

# Convert to list of transactions (list of lists)
transactions = grouped.tolist()

# ---------------------- Analyze Basic Transaction Stats ----------------------

num_transactions = len(transactions)                          # Total number of transactions
print(f"Number of unique transactions: {num_transactions}")

# Compute stats: average, min, and max number of items per transaction
transaction_lengths = [len(t) for t in transactions]
print(f"Average items per transaction: {np.mean(transaction_lengths)}")
print(f"Min items per transaction: {np.min(transaction_lengths)}")
print(f"Max items per transaction: {np.max(transaction_lengths)}")

# ---------------------- Calculate Minimum Support ----------------------

# Use a threshold of 10 transactions to set support
min_support = 10 / num_transactions
print(f"Calculated min_support: {min_support:.6f}")

# ---------------------- Apply Apriori Algorithm ----------------------

rules = apriori(transactions=transactions,
                min_support=min_support,      # Minimum support
                min_confidence=0.2,           # Minimum confidence
                min_lift=1.5,                 # Minimum lift
                min_length=2,                 # Rules must have at least 2 items
                max_length=2)                 # Rules must have at most 2 items

# ---------------------- Process and Display Rules ----------------------

results = list(rules)                                              # Convert generator to list
df_rules = pd.DataFrame(inspect(results), columns=['Buy', 'Get', 'Support'])  # Extract into DataFrame

# Display top 10 rules based on support
print(df_rules.sort_values(by='Support', ascending=False).head(10))
