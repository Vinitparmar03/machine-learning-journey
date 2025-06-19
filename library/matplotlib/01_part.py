# 📦 Imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import random

# ============================
# Scatter Plot
# ============================
X_data = np.random.random(50) * 100
Y_data = np.random.random(50) * 100

# plt.scatter(X_data, Y_data, c="black", s=150, marker="*")
# plt.show()

# ============================
# Line Plot
# ============================
years = [2006 + x for x in range(16)]
weight = [80, 83, 84, 85, 86, 82, 81, 79, 90, 93, 84, 87, 85, 82, 95, 84]

# plt.plot(years, weight, "g--", lw=3)
# plt.show()

# ============================
# Bar Chart
# ============================
x = ["c++", "c#", "java", "python", "go"]
y = [10, 140, 1, 50, 20]

# plt.bar(x, y, color="red", align="edge", width=0.5, edgecolor="green", lw=1)
# plt.show()

# ============================
# Histogram
# ============================
ages = np.random.normal(20, 1.5, 1008)

# plt.hist(ages, bins=20, color="blue", cumulative=True)
# plt.xlabel("Ages")
# plt.ylabel("Count")
# plt.show()

# ============================
# Pie Chart
# ============================
languages = ["Python", "Java", "C++", "C#", "JavaScript"]
popularity = [56, 20, 10, 8, 6]
explodes = (0, 0, 0, 0, 0.1)

# plt.pie(popularity, labels=languages, explode=explodes, autopct="%.2f%%", pctdistance=1.5, startangle=90)
# plt.show()

# ============================
# Boxplot (Heights)
# ============================
heights = np.random.normal(172, 8, 300)

# plt.boxplot(heights)
# plt.show()

# ============================
# Combined Boxplot (4 data segments)
# ============================
first = np.linspace(0, 10, 25)
second = np.linspace(10, 200, 25)
third = np.linspace(200, 210, 25)
fourth = np.linspace(210, 230, 25)

# data = np.concatenate((first, second, third, fourth))
# plt.boxplot(data)
# plt.show()

# ============================
# Line Plot with Y-ticks
# ============================
years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
income = [55, 56, 62, 61, 72, 72, 73, 75]
income_ticks = list(range(50, 81, 2))

# plt.plot(years, income)
# plt.title("Income of John in USD", fontsize=20)
# plt.xlabel("Years")
# plt.ylabel("Yearly Income")
# plt.yticks(income_ticks, [f"${x}k usd" for x in income_ticks])
# plt.show()

# ============================
# Multiple Stocks Line Plot with Legend
# ============================
stock_a = [100, 102, 99, 101, 101, 100, 102]
stock_b = [90, 95, 102, 104, 105, 103, 108]
stock_c = [110, 115, 109, 105, 100, 98, 95]

# plt.plot(stock_a, label="Microsoft")
# plt.plot(stock_b, label="Amazon")
# plt.plot(stock_c, label="Google")
# plt.legend(loc="lower right")
# plt.show()

# ============================
# Pie Chart with Legend
# ============================
votes = [10, 2, 5, 16, 22]
people = ["A", "B", "C", "D", "E"]

# plt.pie(votes, labels=None)
# plt.legend(labels=people)
# plt.show()

# ============================
# Two Separate Figures
# ============================
x1, y1 = np.random.random(100), np.random.random(100)
x2, y2 = np.random.random(100), np.random.random(100)

# plt.figure(1)
# plt.scatter(x1, y1)

# plt.figure(2)
# plt.plot(x2, y2)
# plt.show()

# ============================
# Subplots (2x2 grid)
# ============================
x = np.arange(1, 100)
# fig, axs = plt.subplots(2, 2)

# axs[0, 0].plot(x, np.sin(x))
# axs[0, 0].set_title("Sine Wave")

# axs[0, 1].plot(x, np.cos(x))
# axs[0, 1].set_title("Cos Wave")

# axs[1, 0].plot(x, np.random.random(99))
# axs[1, 0].set_title("Random")

# axs[1, 1].plot(x, np.log(x))
# axs[1, 1].set_title("Log Function")
# axs[1, 1].set_xlabel("X-axis")

# plt.suptitle("Four Plots")
# plt.tight_layout()
# plt.savefig("fourplots.png", dpi=300, transparent=True, bbox_inches="tight")
# plt.show()

# ============================
# 3D Plot (Surface)
# ============================
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x = np.arange(-5, 5, 0.1)
# y = np.arange(-5, 5, 0.1)
# x, y = np.meshgrid(x, y)
# z = np.sin(x) + np.cos(y)

# ax.plot_surface(x, y, z, cmap="Spectral")
# ax.set_title("3D Surface Plot")
# ax.view_init(azim=0, elev=90)
# plt.show()

# ============================
# Live Bar Plot: Coin Toss Simulation
# ============================
heads_tails = [0, 0]

for _ in range(100000):
    heads_tails[random.randint(0, 1)] += 1
    plt.bar(["Heads", "Tails"], heads_tails, color=["red", "blue"])
    plt.pause(0.001)

plt.show()
