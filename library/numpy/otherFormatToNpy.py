import cv2
import numpy as np

# Load the image (as BGR by default)
image = cv2.imread('6493233.jpg')  # Replace with your image path

# Save as .npy
np.save('numpy-logo.npy', image)

# Load back (optional)
loaded_image = np.load('numpy-logo.npy')

# To verify
cv2.imshow("Loaded Image", loaded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
