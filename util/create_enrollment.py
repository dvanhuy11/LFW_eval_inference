import os
import json
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from tqdm import tqdm

def extract_face(filename,detector,required_size=(224, 224)):
    pixels = pyplot.imread(filename)
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def get_embeddings(filenames,detector):
    faces = [extract_face(f,detector) for f in tqdm(filenames, desc="Extracting Faces")]
    samples = asarray(faces, 'float32') 
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    return model.predict(samples, verbose=0)

dataset_directory = '/home/huydinh_intern/lfw'
output_file_name = '/home/huydinh_intern/enrollment_data.json'
enrollment_data = []
detector = MTCNN()
folder_name = [os.path.join(dataset_directory, person) for person in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, person))]
for person_folder in tqdm(folder_name, desc="Processing Person Folders"):
    images = [os.path.join(person_folder, img) for img in os.listdir(person_folder) if img.endswith('.jpg')]
    if images:
        feature = get_embeddings([images[0]],detector)[0].tolist()
        label = os.path.basename(person_folder)
        enrollment_data.append({'feature': feature, 'label': label})

with open(output_file_name, 'w') as f:
    json.dump(enrollment_data, f, indent=4)
