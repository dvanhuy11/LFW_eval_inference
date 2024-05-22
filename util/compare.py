import os
import json
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import torch
from tqdm import tqdm
import numpy as np

def json_numpy_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        features = np.array([np.array(entry['feature']) for entry in data])
        labels = [entry['label'] for entry in data]
    return features, labels

# Function to extract a single face from a photograph
def extract_face(filename, required_size=(224, 224)):
    pixels = pyplot.imread(filename)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# Function to extract faces and calculate face embeddings
def get_embeddings(filenames):
    faces = [extract_face(f) for f in tqdm(filenames, desc="Extracting Faces")]
    samples = asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    return model.predict(samples, verbose=0)

enrollment_features, enrollment_labels = load_data('/home/huydinh_intern/enrollment_data.json')
dataset_directory = '/home/huydinh_intern/lfw'  
output_file_name = '/home/huydinh_intern/LFW_evaluation.json'
results_dict = {}

# Organize dataset for processing
folder_name = sorted([os.path.join(dataset_directory, person) for person in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, person))])

TP, FP, FN, TN = 0, 0, 0, 0
comparison_count = 0 
save_interval = 50
# Perform comparisons
for folder_index, person_folder in enumerate(tqdm(folder_name, desc="Performing Comparisons")):
    images = [os.path.join(person_folder, img) for img in os.listdir(person_folder) if img.endswith('.jpg')]
    if len(images) == 1:
        key = f"{folder_index}_0"
        current_features = get_embeddings([images[0]])[0]
        similarities = [1 - cosine(current_features, ef) for ef in enrollment_features]
        best_match_index = similarities.index(max(similarities))
        best_match_label = enrollment_labels[best_match_index]
        best_match_score = max(similarities)

        results_dict.setdefault(str(folder_index), {})[key] = {
            # "id": best_match_index,
            "name": best_match_label,
            "score": best_match_score
        }

        # Calculate TP and FP based on threshold
        if best_match_score >= 0.55:
            if os.path.basename(person_folder) == best_match_label:
                TP += 1
            else:
                FP += 1
        else:
            if os.path.basename(person_folder) == best_match_label:
                FN += 1
            else:
                TN += 1
    else:
        image_index = 2  # Reset image_index for each new person folder
        for label, image_path in [(os.path.basename(person_folder), img) for img in images[1:]]:
            current_features = get_embeddings([image_path])[0]
            similarities = [1 - cosine(current_features, ef) for ef in enrollment_features]
            best_match_index = similarities.index(max(similarities))
            best_match_label = enrollment_labels[best_match_index]
            key = f"{folder_index}_{image_index}"

            best_match_score = max(similarities)

            results_dict.setdefault(str(folder_index), {})[key] = {
                # "id": best_match_index,
                "name": best_match_label,
                "score": best_match_score
            }

            # Calculate TP and FP based on threshold
            if best_match_score >= 0.55:
                if label == best_match_label:
                    TP += 1
                else:
                    FP += 1
            else:
                if label == best_match_label:
                    FN += 1 
                else:
                    TN += 1

            image_index += 1  # Increment image_index within the inner loop
        comparison_count += 1
        if comparison_count % save_interval == 0:
            with open(output_file_name, 'w') as file:
                json.dump(results_dict, file, indent=4, default=json_numpy_serializable)


print(f"TP: {TP}")
print(f"FP: {FP}")
print(f"TN: {TN}")
print(f"FN: {FN}")

accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0

# Save results and accuracy
results_dict['accuracy'] = accuracy
results_dict['TP'] = TP
results_dict['FP'] = FP
results_dict['TN'] = TN
results_dict['FN'] = FN
with open(output_file_name, 'w') as file:
    json.dump(results_dict, file, indent=4)
print(f"Results have been written to {output_file_name}")
print(f"Accuracy: {accuracy:.2f}")
