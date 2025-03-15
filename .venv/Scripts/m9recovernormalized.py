# m9 recover normalized images to verify

import os
import json
import numpy as np
from PIL import Image

# Define input and output paths
INPUT_FOLDER = f"BillVicarsYoutube/normalized"
INPUT_FILE = "normalized_data.json"
OUTPUT_FOLDER = f"BillVicarsYoutube/reconstructed"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the normalized data from JSON
with open(f"{INPUT_FOLDER}/{INPUT_FILE}", "r") as json_file:
    dataset = json.load(json_file)

# Process each entry in the dataset
for entry in dataset:
    label = entry["label"]  # Get the label (e.g., "A")
    image_data = np.array(entry["image"], dtype=np.float32)  # Convert list to NumPy array

    # Convert from normalized [0,1] back to [0,255]
    image_array = (image_data * 255).astype(np.uint8)  # Ensure uint8 format for RGB

    # Create an image from the array
    image = Image.fromarray(image_array, "RGB")

    # Save the reconstructed image
    output_path = f"{OUTPUT_FOLDER}/{label}.png"
    image.save(output_path)

print(f"Reconstructed images saved in {OUTPUT_FOLDER}")
