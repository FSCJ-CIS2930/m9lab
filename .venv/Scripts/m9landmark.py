# m9 identify landmarks on grayscale images

import os
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# Define input and output paths
INPUT_FOLDER = f"BillVicarsYoutube/grayscale"
OUTPUT_FOLDER = f"BillVicarsYoutube/landmarks"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# Function to detect hand landmarks
def detect_hand_landmarks(image_array):
    """
    Detects hand landmarks in the grayscale image and draws them on the image.
    """
    # Convert grayscale to RGB (MediaPipe requires 3 channels)
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)

    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # Process the image
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(image_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return image_rgb  # Return the image with landmarks drawn


# Process each image in the grayscale folder
for filename in sorted(os.listdir(INPUT_FOLDER)):  # Ensure order (A.png, B.png, etc.)
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = f"{INPUT_FOLDER}/{filename}"

        # Load the grayscale image
        image = Image.open(input_path).convert("L")  # Ensure it's grayscale
        image_array = np.array(image, dtype=np.uint8)  # Convert to NumPy array

        # Detect hand landmarks and draw on the image
        landmark_image = detect_hand_landmarks(image_array)

        # Convert back to PIL Image
        output_image = Image.fromarray(landmark_image)

        # Save the image with hand landmarks
        output_path = f"{OUTPUT_FOLDER}/{filename}"
        output_image.save(output_path)

print(f"Hand landmark images saved in {OUTPUT_FOLDER}")