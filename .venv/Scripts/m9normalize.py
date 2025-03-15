# m9 normalize images

import os
import json
import numpy as np
from PIL import Image

# Define input and output paths
INPUT_FOLDER = f"BillVicarsYoutube/resized"
OUTPUT_FOLDER = f"BillVicarsYoutube/normalized"
OUTPUT_FILE = "normalized_data.json"

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
OUTPUT_JSON_FILE = f"{OUTPUT_FOLDER}/{OUTPUT_FILE}"

# Initialize list to store data
dataset = []


def normalize_image(image_path):
    """Load an image, normalize it to [0,1] range, and return as a list."""
    image = Image.open(image_path).convert("RGB")  # Ensure 3-channel color
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]

    return image_array.tolist()  # Convert NumPy array to list


# Process all images in the input folder
for filename in sorted(os.listdir(INPUT_FOLDER)):  # Sort to maintain order (A, B, C, ...)
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = f"{INPUT_FOLDER}/{filename}"
        image_data = normalize_image(input_path)

        # Extract label from filename (e.g., "A.png" → "A")
        label = os.path.splitext(filename)[0]

        # Append data as a dictionary
        dataset.append({"label": label, "image": image_data})

# Save dataset to JSON
with open(OUTPUT_JSON_FILE, "w") as json_file:
    json.dump(dataset, json_file)

print(f"Normalized data saved to {OUTPUT_JSON_FILE}")
