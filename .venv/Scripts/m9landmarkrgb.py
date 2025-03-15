# m9 identify landmarks on resized RGB images

import os
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np

# Define input and output paths
INPUT_FOLDER = f"BillVicarsYoutube/resized"
OUTPUT_FOLDER = f"BillVicarsYoutube/landmarks_rgb"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to detect and draw hand landmarks on RGB images
def detect_hand_landmarks(image_array):
    """
    Detects hand landmarks in an RGB image and draws them.
    """
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        # Process the image
        results = hands.process(image_array)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(image_array, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return image_array  # Return the image with landmarks drawn

# Process each image in the resized folder
for filename in sorted(os.listdir(INPUT_FOLDER)):  # Ensure order (A.png, B.png, etc.)
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        input_path = f"{INPUT_FOLDER}/{filename}"

        # Load the RGB image
        image = Image.open(input_path).convert("RGB")  # Ensure it's RGB
        image_array = np.array(image)  # Convert to NumPy array

        # Detect hand landmarks and draw on the image
        landmark_image = detect_hand_landmarks(image_array)

        # Convert back to PIL Image
        output_image = Image.fromarray(landmark_image)

        # Save the image with hand landmarks
        output_path = f"{OUTPUT_FOLDER}/{filename}"
        output_image.save(output_path)

print(f"Hand landmark images saved in {OUTPUT_FOLDER}")