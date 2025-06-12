import numpy as np  # NumPy for efficient numerical array operations
import matplotlib.pyplot as plt  # Matplotlib for image visualization

# ------------------ Creating Arrays ------------------

# A simple 1D array
array1 = np.array([1, 2, 3, 4, 5])

# A 3x3 array with random float values between 0 and 1
array2 = np.random.random((3, 3))

# A 4x4 array filled with zeros
array3 = np.zeros((4, 4))

# ------------------ Saving Arrays to .npy Files ------------------

# Save each array to a binary file format (.npy) using NumPy
np.save('array1.npy', array1)
np.save('array2.npy', array2)
np.save('array3.npy', array3)

# ------------------ Loading Saved Arrays ------------------

# Load and display array1 from file
loaded_array = np.load('array1.npy')
print("\n🔹 Loaded array1 (1D array):\n", loaded_array)

# Load and display array2 from file
loaded_array = np.load('array2.npy')
print("\n🔹 Loaded array2 (3x3 random array):\n", loaded_array)

# Load and display array3 from file
loaded_array = np.load('array3.npy')
print("\n🔹 Loaded array3 (4x4 zeros array):\n", loaded_array)

# ------------------ Optional: Display Image (If File Exists) ------------------

try:
    # Try loading an image file named 'numpy-logo.npy' (should contain image pixel data)
    logo = np.load('numpy-logo.npy')
    
    # Create a figure with 2 subplots side by side
    plt.figure(figsize=(8, 6))

    # Original Logo
    plt.subplot(1, 2, 1)
    plt.imshow(logo)  # Display image
    plt.title("Numpy Logo")

    # Dark logo (inverse colors) = subtract pixel values from 1
    dark_logo = 1 - logo
    plt.subplot(1, 2, 2)
    plt.imshow(dark_logo)
    plt.title("Numpy Dark Logo")

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("\n⚠️ Numpy logo file not found. Skipping image visualization.")
