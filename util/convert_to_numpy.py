import os
import json
import numpy as np
from PIL import Image
from numpy import asarray
from tqdm import tqdm
import cv2
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

# Load the ultra-light model for face detection
net = cv2.dnn.readNetFromONNX("/media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/version-RFB-320.onnx")

def extract_face(filename, required_size=(224, 224)):
    image = cv2.imread(filename)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (320, 240), (104.0, 177.0, 123.0), swapRB=False)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[1]):
        confidence = detections[0, i, 2]
        if confidence > 0.5:
            x1, y1, x2, y2 = int(detections[0, i, 0] * w), int(detections[0, i, 1] * h), \
                             int((detections[0, i, 0] + detections[0, i, 2]) * w), int((detections[0, i, 1] + detections[0, i, 3]) * h)
            face = image[y1:y2, x1:x2]
            face_image = Image.fromarray(face)
            face_image = face_image.resize(required_size)
            face_array = asarray(face_image)
            return face_array
    return None

def get_embeddings(filenames):
    faces = [extract_face(f) for f in tqdm(filenames, desc="Extracting Faces")]
    if faces:
        samples = asarray(faces, 'float32')
        samples = preprocess_input(samples, version=2)
        model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        return model.predict(samples)
    return []

dataset_directory = '/media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/test'  
enrollment_file = '/media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/output/enrollment.npy'
labels_file = '/media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/output/labels.json'

person_folders = [os.path.join(dataset_directory, person) for person in tqdm(os.listdir(dataset_directory), desc="Reading Person Folders") if os.path.isdir(os.path.join(dataset_directory, person))]
enrollment_features = []
enrollment_labels = []

for person_folder in tqdm(person_folders, desc="Processing Person Folders"):
    images = [os.path.join(person_folder, img) for img in os.listdir(person_folder) if img.endswith('.jpg')]
    if images:
        embedding = get_embeddings([images[0]])
        if embedding is not None:
            enrollment_features.append(embedding[0])
            enrollment_labels.append(os.path.basename(person_folder))

np.save(enrollment_file, np.array(enrollment_features))
with open(labels_file, 'w') as file:
    json.dump(enrollment_labels, file)

print("Enrollment data and labels have been saved.")
