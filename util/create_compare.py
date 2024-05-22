import os
import json
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import tensorflow as tf
import torch


#-----------------------------------------------------------------------------------------------
# |  pip install tensorflow==2.12.1                                                            |
# |  pip install keras==2.12                                                                   |
# |                                                                                            |
# |  I solved this issue by changing the import from                                           |
# |     from keras.engine.topology import get_source_inputs                                    |
# |  to                                                                                        |
# |      from keras.utils.layer_utils import get_source_inputs in keras_vggface/models.py.     |
#-----------------------------------------------------------------------------------------------


# Function to extract a single face from a photograph
def extract_face(filename, required_size=(460, 259)):
    pixels = pyplot.imread(filename)
    detector = MTCNN()
    print("detect done")
    results = detector.detect_faces(pixels)
    print("result")
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    print("resize")
    face_array = asarray(image)
    print("extract face done----------------")
    return face_array

# Function to extract faces and calculate face embeddings
def get_embeddings(filenames):
    faces = [extract_face(f) for f in filenames]
    print("faces DONE")
    samples = asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50', include_top=False, input_shape=(259, 460, 3), pooling='avg')
    print("get embedding done ")
    return model.predict(samples)

dataset_directory = '/media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/test'  
output_file_name = '/media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/output/LFW_evaluation.json'
results_dict = {}

# Organize dataset for processing
person_folders = [os.path.join(dataset_directory, person) for person in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, person))]
enrollment_features = []
enrollment_labels = []
comparison_images = []

# Create the enrollment set
for person_folder in person_folders:
    images = [os.path.join(person_folder, img) for img in os.listdir(person_folder) if img.endswith('.jpg')]
    if images:
        enrollment_features.append(get_embeddings([images[0]])[0])  # Add first image's embedding
        enrollment_labels.append(os.path.basename(person_folder))
        if len(images) > 1:
            comparison_images.extend([(os.path.basename(person_folder), img) for img in images[1:]])

TP = 0
FP = 0

# Perform comparisons
for label, image_path in comparison_images:
    current_features = get_embeddings([image_path])[0]
    similarities = [1 - cosine(current_features, ef) for ef in enrollment_features]
    best_match_index = similarities.index(max(similarities))
    best_match_label = enrollment_labels[best_match_index]
    folder_index = enrollment_labels.index(label) + 1  # Get index for unique key generation
    image_index = comparison_images.index((label, image_path)) + 2  # Assuming index starts from 2
    key = f"{folder_index}_{image_index}"

    best_match_score = max(similarities)

    results_dict.setdefault(str(folder_index), {})[key] = {
        "id": best_match_index,
        "name": best_match_label,
        "score": best_match_score
    }

    # Calculate TP and FP based on threshold
    if best_match_score >= 0.55:
        if label == best_match_label:
            TP += 1
        else:
            FP += 1

# Calculate accuracy
total_predictions = TP + FP
accuracy = TP / total_predictions if total_predictions > 0 else 0

# Save results and accuracy
results_dict['accuracy'] = accuracy
with open(output_file_name, 'w') as file:
    json.dump(results_dict, file, indent=4)

print(f"Results have been written to {output_file_name}")
print(f"Accuracy: {accuracy:.2f}")
