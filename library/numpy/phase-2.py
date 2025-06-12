# ------------------------ Imports ------------------------
import numpy as np  # Importing NumPy for numerical array operations

# ------------------------ 1D Array Slicing ------------------------

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Basic slicing: Extract elements from index 2 to 6
print("Basic Slicing [2:7]:", arr[2:7])  # Output: [3 4 5 6 7]

# Slicing with steps: From index 1 to 7, pick every second element
print("Basic Slicing with steps [1:8:2]:", arr[1:8:2])  # Output: [2 4 6 8]

# Negative indexing: Fetch 3rd element from end
print("Negative indexing [-3]:", arr[-3])  # Output: 8

# ------------------------ 2D Array Indexing ------------------------

arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Accessing a specific element [row=1, col=2]
print("Specific element [1,2]:", arr_2d[1, 2])  # Output: 6

# Accessing entire row 1
print("Entire row [1]:", arr_2d[1])  # Output: [4 5 6]

# Accessing entire column 1
print("Entire column [:,1]:", arr_2d[:, 1])  # Output: [2 5 8]

# ------------------------ Sorting ------------------------

unsorted = np.array([3, 5, 2, 6, 8, 4, 3, 7])

# Sorting a 1D array
print("Sorted 1D Array:", np.sort(unsorted))  # Output: [2 3 3 4 5 6 7 8]

arr_2d_unsorted = np.array([[3, 1],
                            [1, 2],
                            [4, 3]])

# Sorting by column (axis=0)
print("Sorted 2D array by column (axis=0):\n", np.sort(arr_2d_unsorted, axis=0))

# Sorting by row (axis=1)
print("Sorted 2D array by row (axis=1):\n", np.sort(arr_2d_unsorted, axis=1))

# ------------------------ Filtering ------------------------

numbers = np.array([1,2,3,4,5,6,7,8,9,10])

# Filtering even numbers using boolean indexing
even_number = numbers[numbers % 2 == 0]
print("Even numbers:", even_number)

# Filtering with a mask: numbers greater than 5
mask = numbers > 5
print("Numbers greater than 5 (mask):", numbers[mask])

# ------------------------ Fancy Indexing & np.where ------------------------

# Fancy indexing: Fetch specific indices
indices = [0, 2, 4]
print("Fancy indexing [0,2,4]:", numbers[indices])  # Output: [1 3 5]

# np.where() to find indices of condition
where_result = np.where(numbers > 5)
print("np.where result (indices where >5):", where_result)
print("Elements where condition is true:", numbers[where_result])  # Output: [6 7 8 9 10]

# np.where() to conditionally modify array (if >5 then x4, else keep same)
condition_array = np.where(numbers > 5, numbers * 4, numbers)
print("Condition-based transformation (if >5, multiply by 4):", condition_array)

# np.where() to return strings based on condition
condition_array_str = np.where(numbers > 5, "true", "false")
print("Condition array with 'true'/'false':", condition_array_str)

# ------------------------ Adding & Combining Arrays ------------------------

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Element-wise addition
addition = arr1 + arr2
print("Addition of two arrays:", addition)

# Concatenation of two arrays
combined = np.concatenate((arr1, arr2))
print("Combined array:", combined)

# ------------------------ Stacking Rows & Columns ------------------------

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([7, 8, 9])

# Checking shape compatibility
print("Are a and b shape compatible for operations?", a.shape == b.shape)

# Vertical stacking (adding row)
original = np.array([[1, 2],
                     [3, 4]])
new_row = np.array([[5, 6]])
with_new_row = np.vstack((original, new_row))
print("Original matrix:\n", original)
print("Matrix after adding a new row:\n", with_new_row)

# Horizontal stacking (adding column)
new_col = np.array([[7], [8]])
with_new_col = np.hstack((original, new_col))
print("Matrix after adding a new column:\n", with_new_col)

# ------------------------ Deleting Data ------------------------

arr = np.array([1, 2, 3, 4, 5])

# Deleting index 2 (3rd element)
deleted = np.delete(arr, 2)
print("1D array after deleting index 2:", deleted)  # Output: [1 2 4 5]

# Deleting a row from a 2D array
matrix_2d = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
deleted_row = np.delete(matrix_2d, 1, axis=0)
print("2D Array after deleting row index 1:\n", deleted_row)

# Deleting a column from a 2D array
deleted_col = np.delete(matrix_2d, 1, axis=1)
print("2D Array after deleting column index 1:\n", deleted_col)
