import os
import json
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
from tqdm import tqdm

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

def extract_face(filename,detector, required_size=(224, 224)):
    pixels = pyplot.imread(filename)
    results = detector.detect_faces(pixels)
    detector = None  
    if results:
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        return face_array
    return None

def get_embeddings(faces,model):
    samples = asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    return model.predict(samples, verbose=0)

def save_results(metrics, comparisons, output_file_name):
    results = {
        "accuracy": metrics['accuracy'],
        "TP": metrics['TP'],
        "FP": metrics['FP'],
        "TN": metrics['TN'],
        "FN": metrics['FN']
    }
    results.update(comparisons)
    with open(output_file_name, 'w') as file:
        json.dump(results, file, indent=4, default=json_numpy_serializable)
#load JSON
enrollment_features, enrollment_labels = load_data('/media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/output/enrollment_data.json')
dataset_directory = '/media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/lfw'
output_file_name = '/media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/LFW_evaluate/output/LFW_evaluation.json'

#load model
detector = MTCNN()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

results_dict = {}
metrics = {"accuracy": 0, "TP": 0, "FP": 0, "TN": 0, "FN": 0}
comparisons = {}
folder_name = sorted([os.path.join(dataset_directory, person) for person in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, person))])

batch_size = 5  # Adjust batch size based on your system's capability
for i in tqdm(range(0, len(folder_name), batch_size), desc="Processing batches"):
    batch_folders = folder_name[i:i+batch_size]
    all_faces = []
    image_paths = []
    for person_folder in batch_folders:
        images = [os.path.join(person_folder, img) for img in os.listdir(person_folder) if img.endswith('.jpg')]
        for image_path in images:
            face = extract_face(image_path,detector)
            if face is not None:
                all_faces.append(face)
                image_paths.append(image_path)

    # Calculate embeddings and perform comparison for each batch
    embeddings = get_embeddings(all_faces,model)

    for idx, (face_embedding, image_path) in enumerate(zip(embeddings, image_paths)):
        similarities = [1 - cosine(face_embedding, ef) for ef in enrollment_features]
        best_match_index = similarities.index(max(similarities))
        best_match_label = enrollment_labels[best_match_index]
        best_match_score = max(similarities)

        person_folder = os.path.dirname(image_path)
        folder_index = folder_name.index(person_folder)
        image_index = idx

        key = f"{folder_index}_{image_index}"
        comparisons.setdefault(str(folder_index), {})[key] = {
            "name": best_match_label,
            "score": best_match_score
        }

        # Calculate TP, FP, TN, FN based on threshold
        if best_match_score >= 0.55:
            if os.path.basename(person_folder) == best_match_label:
                metrics['TP'] += 1
            else:
                metrics['FP'] += 1
        else:
            if os.path.basename(person_folder) == best_match_label:
                metrics['FN'] += 1
            else:
                metrics['TN'] += 1

    accuracy = (metrics['TP'] + metrics['TN']) / (metrics['TP'] + metrics['TN'] + metrics['FP'] + metrics['FN']) if (metrics['TP'] + metrics['TN'] + metrics['FP'] + metrics['FN']) != 0 else 0
    metrics['accuracy'] = accuracy

save_results(metrics, comparisons, output_file_name)
print(f"Results have been written to {output_file_name}")
