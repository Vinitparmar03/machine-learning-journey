# ------------------ Import Required Libraries ------------------
import numpy as np
import pandas as pd
from apyori import apriori

# ------------------ Load and Preprocess Dataset ------------------
# Load the Groceries dataset
df = pd.read_csv('Groceries_dataset.csv')

# Create a unique transaction ID by combining 'Member_number' and 'Date'
df['Transaction'] = df['Member_number'].astype(str) + '_' + df['Date']

# Group items by transaction ID
grouped = df.groupby('Transaction')['itemDescription'].apply(list)

# Convert grouped items into a list of transactions
transactions = grouped.tolist()

# ------------------ Basic Transaction Stats ------------------
# Total number of transactions
num_transactions = len(transactions)
print(f"Number of unique transactions: {num_transactions}")

# Calculate and display item count per transaction
transaction_lengths = [len(t) for t in transactions]
print(f"Average items per transaction: {np.mean(transaction_lengths):.2f}")
print(f"Min items per transaction: {np.min(transaction_lengths)}")
print(f"Max items per transaction: {np.max(transaction_lengths)}")

# ------------------ Calculate Minimum Support ------------------
# We want items with at least 10 occurrences
min_support = 10 / num_transactions
print(f"Calculated min_support: {min_support:.6f}")

# ------------------ Apply Apriori Algorithm ------------------
rules = apriori(transactions=transactions,
                min_support=min_support,
                min_confidence=0.2,   # Minimum confidence threshold
                min_lift=1.5,         # Minimum lift value
                min_length=2,         # At least 2 items in the rule
                max_length=2)         # No more than 2 items in the rule

# ------------------ Rule Extraction Function ------------------
# This function extracts useful metrics from the rule objects
def inspect(results):
    lhs = [tuple(r.ordered_statistics[0].items_base)[0] for r in results]
    rhs = [tuple(r.ordered_statistics[0].items_add)[0] for r in results]
    support = [r.support for r in results]
    confidence = [r.ordered_statistics[0].confidence for r in results]
    lift = [r.ordered_statistics[0].lift for r in results]
    return list(zip(lhs, rhs, support, confidence, lift))

# ------------------ Transform and Display Rules ------------------
# Convert the rules to a DataFrame
results = list(rules)
df_rules = pd.DataFrame(inspect(results), columns=['Buy', 'Get', 'Support', 'Confidence', 'Lift'])

# Show top 10 rules sorted by Lift
print(df_rules.sort_values(by='Lift', ascending=False).head(10))
