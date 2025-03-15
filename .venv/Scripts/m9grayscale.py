# m9 convert normalized RGB to grayscale

import os
import json
import numpy as np
from PIL import Image

# Define input and output paths
INPUT_FOLDER = f"BillVicarsYoutube/normalized"
INPUT_FILE = "normalized_data.json"
OUTPUT_FOLDER = f"BillVicarsYoutube/grayscale"

# Construct full input file path
INPUT_JSON_FILE = f"{INPUT_FOLDER}/{INPUT_FILE}"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the normalized data from JSON
with open(INPUT_JSON_FILE, "r") as json_file:
    dataset = json.load(json_file)

# Function to convert RGB to grayscale
def rgb_to_grayscale(image_array):
    """
    Convert an RGB image array (normalized 0-1) to grayscale using the standard formula:
        Grayscale = 0.299 * R + 0.587 * G + 0.114 * B
    """
    image_array = np.array(image_array, dtype=np.float32)  # Convert list to NumPy array
    grayscale_array = np.dot(image_array, [0.299, 0.587, 0.114])  # Apply conversion formula
    return grayscale_array

# Process each entry in the dataset
for entry in dataset:
    label = entry["label"]  # Get the label (e.g., "A")
    image_data = np.array(entry["image"], dtype=np.float32)  # Convert list to NumPy array

    # Convert to grayscale
    grayscale_data = rgb_to_grayscale(image_data)

    # Convert back to 0-255 range
    grayscale_array = (grayscale_data * 255).astype(np.uint8)  # Ensure uint8 format

    # Create an image from the array
    grayscale_image = Image.fromarray(grayscale_array, "L")  # "L" mode for grayscale

    # Save the grayscale image
    output_path = f"{OUTPUT_FOLDER}/{label}.png"
    grayscale_image.save(output_path)

print(f"Grayscale images saved in {OUTPUT_FOLDER}")
