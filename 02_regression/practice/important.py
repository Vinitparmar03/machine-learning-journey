import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset (make sure 'winequality.csv' is in your working directory)
df = pd.read_csv("winequality.csv")

# Step 2: Check if there are any missing values anywhere in the dataset
print("Any missing values in dataset?:", df.isnull().values.any())

# Step 3: Count missing values in each column
print("\nMissing values per column:")
print(df.isnull().sum())

# Step 4: Total number of missing values in the entire dataset
print("\nTotal missing values in dataset:", df.isnull().sum().sum())

# Step 5: Percentage of missing values per column
print("\nPercentage of missing values per column:")
print((df.isnull().sum() / len(df)) * 100)

# Step 6: Display rows that contain any missing values
print("\nRows with missing values:")
print(df[df.isnull().any(axis=1)])

# Step 7: Visualize missing values using a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()
