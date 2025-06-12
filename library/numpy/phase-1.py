# ------------------------ Imports ------------------------
import numpy as np      # Importing NumPy library for numerical operations
import time             # Importing time library to measure execution time

# -------------------- Array Creation ---------------------

# Creating a 1D NumPy array
arr_1d = np.array([1, 2, 3, 4, 5])
print("1D Array:\n", arr_1d)

# Creating a 2D NumPy array (matrix)
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D Array:\n", arr_2d)

# ---------------- Python List vs NumPy Array Multiplication ----------------

# Multiplying a regular Python list (replication, not element-wise)
py_list = [1, 2, 3]
print("\nPython list multiplied by 2:\n", py_list * 2)  # Output: [1, 2, 3, 1, 2, 3]

# Multiplying a NumPy array (element-wise multiplication)
np_array = np.array([1, 2, 3])
print("\nNumPy array multiplied by 2 (element-wise):\n", np_array * 2)  # Output: [2, 4, 6]

# ---------------- Performance Comparison ----------------

# Measuring execution time of list comprehension
start = time.time()
list_comp = [i * 2 for i in range(1000000)]
print("\nTime taken by Python list operation: {:.5f} seconds".format(time.time() - start))

# Measuring execution time of NumPy array operation
start = time.time()
np_array = np.arange(1000000) * 2
print("Time taken by NumPy array operation: {:.5f} seconds".format(time.time() - start))

# -------------------- Special Arrays ---------------------

# Creating an array filled with zeros
zeros = np.zeros((3, 4))
print("\nZeros Array (3x4):\n", zeros)

# Creating an array filled with ones
ones = np.ones((2, 3))
print("\nOnes Array (2x3):\n", ones)

# Creating an array filled with a specific value (10)
full = np.full((2, 2), 10)
print("\nFull Array (2x2 with 10):\n", full)

# Creating a 2x3 array with random floats in [0.0, 1.0)
random = np.random.random((2, 3))
print("\nRandom Array (2x3):\n", random)

# Creating a sequence from 0 to 10 with step size 2
sequence = np.arange(0, 11, 2)
print("\nSequence Array (0 to 10 with step 2):\n", sequence)

# ----------------- Array Dimensions ----------------------

# Creating basic vector (1D)
vector = np.array([1, 2, 3])
print("\nVector (1D):\n", vector)

# Creating a matrix (2D)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("\nMatrix (2D):\n", matrix)

# Creating a tensor (3D)
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\nTensor (3D):\n", tensor)

# ----------------- Array Attributes ----------------------

arr = np.array([[1, 2, 3], [4, 5, 6]])
print("\nArray Shape (rows, columns):", arr.shape)
print("Number of Dimensions:", arr.ndim)
print("Total Number of Elements:", arr.size)
print("Data Type of Elements:", arr.dtype)

# -------------------- Reshaping Arrays -------------------

# Creating a 1D array with 12 elements
arr = np.arange(12)
print("\nOriginal 1D Array:\n", arr)

# Reshaping it to 3 rows and 4 columns
reshaped = arr.reshape((3, 4))
print("\nReshaped Array (3x4):\n", reshaped)

# Flattening reshaped array to 1D using `flatten()` (returns a copy)
flattened = reshaped.flatten()
print("\nFlattened Array (using flatten()):\n", flattened)

# Flattening reshaped array using `ravel()` (returns a view)
raveled = reshaped.ravel()
print("\nRaveled Array (using ravel()):\n", raveled)

# Transposing the reshaped array (rows become columns)
transpose = reshaped.T
print("\nTransposed Array:\n", transpose)
