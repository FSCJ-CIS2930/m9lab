
# Module 9 resize images

# Pillow is the standard library for image processing in Python.
# PIL is a component in that library
from PIL import Image
import os

# Define the target size (standardized for ML models)
TARGET_SIZE = (224, 224)
# Define input and output paths
INPUT_FOLDER = f"BillVicarsYoutube"
OUTPUT_FOLDER = f"BillVicarsYoutube/resized"

# Ensure the output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def resize_image(image_path, output_path):
    image = Image.open(image_path)

    # Resize using "fit within" strategy while maintaining aspect ratio
    image.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)

    # Create a new blank (white) image of the target size
    resized_image = Image.new("RGB", TARGET_SIZE, (255, 255, 255))

    # Paste the resized image at the center
    x_offset = (TARGET_SIZE[0] - image.width) // 2
    y_offset = (TARGET_SIZE[1] - image.height) // 2
    resized_image.paste(image, (x_offset, y_offset))

    # Save the resized image
    resized_image.save(output_path)

# Process all images in the input folder
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = f"{INPUT_FOLDER}/{filename}"
        output_path = f"{OUTPUT_FOLDER}/{filename}"
        resize_image(input_path, output_path)
        print(f"Resized and saved: {output_path}")

print("All images have been resized successfully.")