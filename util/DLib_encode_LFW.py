from imutils import paths
import argparse
import pickle
import cv2
import os
import face_recognition
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to the directory of faces and images")
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encoding")
ap.add_argument("-d", "--detection_method", type=str, default="cnn", help="face detector to use: cnn or hog")
args = vars(ap.parse_args())

# Quantifying faces
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
imagePaths.sort()  # Make sure paths are sorted to maintain order

# Initialize lists and dict
knownEncodings = []
knownNames = []
processed_names = set()  # To keep track of processed individuals

# Process each image
for (i, imagePath) in enumerate(imagePaths):
    name = imagePath.split(os.path.sep)[-2]
    if name in processed_names:
        continue  # Skip if we've already processed this person

    print("[INFO] processing image {}/{}: {}".format(i + 1, len(imagePaths), imagePath))
    processed_names.add(name)

    # Load image and convert from BGR to RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)

    # Save the encoding and name
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# Serialize the facial encodings and names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open(args["encodings"], "wb") as f:
    f.write(pickle.dumps(data))
