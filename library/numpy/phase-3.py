# ------------------------ Imports ------------------------
import numpy as np  # NumPy for data manipulation
import matplotlib.pyplot as plt  # Matplotlib for visualization

# ------------------------ Sales Dataset ------------------------

# Each row = [restaurant_id, sales in 2021, 2022, 2023, 2024]
sales_data = np.array([
    [1, 150000, 100000, 220000, 250000],  # Paradise Biryani
    [2, 120000, 140000, 160000, 190000],  # Baba Ram Dev
    [3, 200000, 230000, 260000, 300000],  # Total Eating
    [4, 100000, 210000, 240000, 270000],  # 90's Food
    [5, 160000, 185000, 205000, 230000],  # Chai Wai
])

print("**** Zomato Sales Analysis ****")

# ------------------------ Basic Info ------------------------

# Displaying the shape of dataset (5 restaurants, 5 columns)
print("\nShape of sales data:", sales_data.shape)

# Showing first 3 restaurant entries as sample
print("\nSample data for 3 restaurants:\n", sales_data[:3])

# ------------------------ Total Sales Per Year ------------------------

# Sum across restaurants for each column (including restaurant_id)
print("\nTotal per column (including ID):", np.sum(sales_data, axis=0))

# Only for years (columns 1 to 4)
yearly_total = np.sum(sales_data[:, 1:], axis=0)
print("Yearly Total Sales (2021–2024):", yearly_total)

# ------------------------ Total Revenue ------------------------

total_revenue = np.sum(yearly_total)
print("Total Revenue from 2021 to 2024:", total_revenue)

# ------------------------ Minimum Sales Per Restaurant ------------------------

# Finding min sale year-wise per restaurant
min_sales = np.min(sales_data[:, 1:], axis=1)
print("Minimum sales per restaurant:", min_sales)

# ------------------------ Maximum Sales Per Year ------------------------

max_sales_per_year = np.max(sales_data[:, 1:], axis=0)
print("Maximum sales per year (2021–2024):", max_sales_per_year)

# ------------------------ Average Sales Per Restaurant ------------------------

avg_sales = np.mean(sales_data[:, 1:], axis=1)
print("Average sales per restaurant (across 4 years):", avg_sales)

# ------------------------ Cumulative Sum ------------------------

cumsum = np.cumsum(sales_data[:, 1:], axis=1)
print("Cumulative sales per restaurant (across years):\n", cumsum)

# ------------------------ Plotting Avg. Cumulative Sales ------------------------

plt.figure(figsize=(10, 6))

# Plot average cumulative sales for all restaurants
plt.plot(np.mean(cumsum, axis=0), marker='o', color='blue')

plt.title("Average Cumulative Sales per Year Across All Restaurants")
plt.xlabel("Years [2021 to 2024]")
plt.ylabel("Average Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------ Vector Math ------------------------

vector1 = np.array([1, 2, 3, 4, 5])
vector2 = np.array([6, 7, 8, 9, 10])

# Element-wise addition
print("Vector Addition:", vector1 + vector2)

# Element-wise multiplication
print("Vector Multiplication:", vector1 * vector2)

# Dot product
print("Dot Product:", np.dot(vector1, vector2))

# Angle (in radians) between two vectors using cosine formula
angle = np.arccos(
    np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
)
print("Angle between vectors (in radians):", angle)

# ------------------------ Vectorized String Operation ------------------------

restaurent_types = np.array(['briyani', 'chinese', 'italian', 'indian', 'gujrati'])

# Applying upper-case transformation using vectorized function
vectorized_upper = np.vectorize(str.upper)
print("Restaurant types in uppercase:", vectorized_upper(restaurent_types))

# ------------------------ Monthly Sales Average ------------------------

# Divide yearly sales by 12 months for each restaurant
monthly_avg = sales_data[:, 1:] / 12
print("Monthly average sales (in thousands):\n", monthly_avg)
