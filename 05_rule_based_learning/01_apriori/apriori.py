# -------------------------- Imports --------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# -------------------- Function to Inspect Rules --------------------
def inspect(results):
    lhs         = [tuple(result.ordered_statistics[0].items_base)[0] for result in results]
    rhs         = [tuple(result.ordered_statistics[0].items_add)[0] for result in results]
    supports    = [result.support for result in results]
    confidences = [result.ordered_statistics[0].confidence for result in results]
    lifts       = [result.ordered_statistics[0].lift for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

# ------------------- Load and Preprocess Dataset -------------------
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20) if str(dataset.values[i, j]) != 'nan'])

# ------------------- Train Apriori Model -------------------
rules = apriori(transactions=transactions, 
                min_support=0.003, 
                min_confidence=0.2, 
                min_lift=3, 
                min_length=2, 
                max_length=2)

# ------------------- Process Results -------------------
results = list(rules)
resultsinDataFrame = pd.DataFrame(inspect(results), 
                                  columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

# ------------------- Display All Rules -------------------
print(resultsinDataFrame)

# ------------------- Display Top 10 Rules by Lift -------------------
print("\nTop 10 Rules Sorted by Lift:")
print(resultsinDataFrame.nlargest(n=10, columns="Lift"))
