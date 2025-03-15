# m9 shear images

import os
import numpy as np
import cv2
from PIL import Image

# Define input and output paths
INPUT_FOLDER = f"BillVicarsYoutube/grayscale"
OUTPUT_FOLDER = f"BillVicarsYoutube/sheared"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to apply shearing to a grayscale image
def apply_shear(image_array, shear_factor=0.3):
    """
    Apply an affine shear transformation to a grayscale image.
    """
    height, width = image_array.shape  # Grayscale images are 2D
    shear_matrix = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
    sheared_image = cv2.warpAffine(image_array, shear_matrix, (width, height))
    return sheared_image

# Process each image in the grayscale folder
for filename in sorted(os.listdir(INPUT_FOLDER)):  # Ensure order (A.png, B.png, etc.)
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = f"{INPUT_FOLDER}/{filename}"

        # Load the grayscale image
        image = Image.open(input_path).convert("L")  # Ensure it's grayscale
        image_array = np.array(image, dtype=np.uint8)  # Convert to NumPy array

        # Apply shearing
        sheared_image = apply_shear(image_array)

        # Convert back to PIL Image (grayscale mode "L")
        augmented_image = Image.fromarray(sheared_image, "L")

        # Save the sheared image in the output folder
        output_path = f"{OUTPUT_FOLDER}/{filename}"
        augmented_image.save(output_path)

print(f"Sheared grayscale images saved in {OUTPUT_FOLDER}")
