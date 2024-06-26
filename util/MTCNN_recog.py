import cv2
from mtcnn.mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from PIL import Image
import numpy as np
import json
from scipy.spatial.distance import cosine
import time

def load_face_data(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)

def extract_embeddings(face_pixels, model):
    face_pixels = face_pixels.astype('float32')
    samples = np.expand_dims(face_pixels, axis=0)
    samples = preprocess_input(samples, version=2)
    return model.predict(samples)[0]

def find_match(known_embeddings, candidate_embedding, threshold=0.5):
    min_dist = float('inf')
    best_match = None
    for item in known_embeddings:
        dist = cosine(item['feature'], candidate_embedding)
        if dist < min_dist:
            min_dist = dist
            best_match = item['label'] if dist < threshold else 'Unknown'
    return best_match

detector = MTCNN(steps_threshold=[0.5, 0.6, 0.7])  

model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
registered_faces = load_face_data('/media/divhuy/63ED6D5823380FB4/HUTECH/TTTN/w1/Inference/output/MTCNN_data.json')

cap = cv2.VideoCapture(0)
frame_count = 0
skip_frames = 10  # Process every 5th frame
face_cache = {}
cache_timeout = 5.5  # Cache timeout in frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_frames != 0:
        continue  # Skip processing for this frame

    frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Reduce frame size
    rgb_frame = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    
    start_time = time.time()
    if face_cache.get("timeout", 0) > 0:
        face_cache["timeout"] -= 1
        faces = face_cache["faces"]
    else:
        faces = detector.detect_faces(rgb_frame)
        face_cache["faces"] = faces
        face_cache["timeout"] = cache_timeout
    
    for face in faces:
        x, y, width, height = face['box']
        x1, y1 = max(0, x * 2), max(0, y * 2)  # Adjust coordinates for original frame size
        face_image = frame[y1:y1 + height * 2, x1:x1 + width * 2]
        face_image = Image.fromarray(face_image)
        face_image = face_image.resize((224, 224))
        face_array = np.asarray(face_image)

        embedding = extract_embeddings(face_array, model)
        label = find_match(registered_faces, embedding)

        cv2.rectangle(frame, (x1, y1), (x1 + width * 2, y1 + height * 2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    end_time = time.time()
    fps_text = f"FPS: {1.0 / (end_time - start_time):.2f}"
    cv2.putText(frame, fps_text, (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)
    
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
