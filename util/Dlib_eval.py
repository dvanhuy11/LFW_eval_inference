import argparse
import pickle
import cv2
import os
import face_recognition
import numpy as np
import json
from scipy.spatial.distance import cosine
from imutils import paths

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to the directory of faces and images")
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encoding")
ap.add_argument("-o", "--output", required=True, help="path to the output JSON file for results")
ap.add_argument("-d", "--detection_method", type=str, default="cnn", help="face detector to use: cnn or hog")
args = vars(ap.parse_args())

# Load the known encodings and names
print("[INFO] loading encodings...")
with open(args["encodings"], "rb") as f:
    data = pickle.load(f)
knownEncodings = data["encodings"]
knownNames = data["names"]

# Prepare to quantify faces
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
imagePaths.sort()  # Make sure paths are sorted to maintain order

# Initialize results and metrics
metrics = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0, 'accuracy': 0}
comparisons = {}

# Prepare dictionary to track indices for each folder
folder_indices = {}
folder_mapping = {}
folder_counter = 0

# Process and compare each image
for imagePath in imagePaths:
    print("[INFO] processing and comparing image {}".format(imagePath))
    person_folder = os.path.basename(os.path.dirname(imagePath))

    # Map folder name to a numeric index
    if person_folder not in folder_mapping:
        folder_mapping[person_folder] = folder_counter
        folder_counter += 1

    folder_index = folder_mapping[person_folder]

    # Initialize index for the person if not already
    if folder_index not in folder_indices:
        folder_indices[folder_index] = 0

    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    test_encodings = face_recognition.face_encodings(rgb, boxes)

    # Check and initialize person folder in comparisons
    if folder_index not in comparisons:
        comparisons[folder_index] = {}

    # For each face detected
    for test_encoding in test_encodings:
        # Compute similarities with known encodings
        similarities = [1 - cosine(test_encoding, ef) for ef in knownEncodings]
        best_match_index = similarities.index(max(similarities))
        best_match_name = knownNames[best_match_index]
        best_match_score = max(similarities)

        # Create a unique key for each image
        image_index = folder_indices[folder_index]
        folder_indices[folder_index] += 1  # Increment the index for this person's folder

        key = f"{folder_index}_{image_index}"
        comparisons[folder_index][key] = {
            "name": best_match_name,
            "score": best_match_score
        }

        # Calculate TP, FP, TN, FN based on a threshold
        threshold = 0.55
        if best_match_score >= threshold:
            if person_folder == best_match_name:
                metrics['TP'] += 1
            else:
                metrics['FP'] += 1
        else:
            if person_folder == best_match_name:
                metrics['FN'] += 1
            else:
                metrics['TN'] += 1

# Calculate accuracy
total = metrics['TP'] + metrics['TN'] + metrics['FP'] + metrics['FN']
metrics['accuracy'] = (metrics['TP'] + metrics['TN']) / total if total > 0 else 0
result = {"metrics": metrics, "comparisons": comparisons}

# Write the results to the specified JSON file
with open(args["output"], "w") as f:
    json.dump(result, f, indent=4)

print("[INFO] Results have been written to {}".format(args["output"]))


