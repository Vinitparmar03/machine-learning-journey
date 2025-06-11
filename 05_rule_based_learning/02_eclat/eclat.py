# -------------------- IMPORTING LIBRARIES --------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori

# -------------------- FUNCTION DEFINITIONS --------------------
def inspect(results):
    lhs         = [tuple(result.ordered_statistics[0].items_base)[0] for result in results]
    rhs         = [tuple(result.ordered_statistics[0].items_add)[0] for result in results]
    supports    = [result.support for result in results]
    return list(zip(lhs, rhs, supports))

# -------------------- DATA PREPROCESSING --------------------
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

transactions = []
for i in range(0, len(dataset)):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20) if str(dataset.values[i, j]) != 'nan'])

# -------------------- TRAINING APRIORI MODEL --------------------
rules = apriori(transactions=transactions, 
                min_support=0.003, 
                min_confidence=0.2, 
                min_lift=3, 
                min_length=2, 
                max_length=2)

# -------------------- CONVERT RESULTS --------------------
results = list(rules)
results_in_df = pd.DataFrame(inspect(results), columns=['Product 1', 'Product 2', 'Support'])

# -------------------- DISPLAYING TOP 10 RESULTS --------------------
print(results_in_df.nlargest(n=10, columns='Support'))
