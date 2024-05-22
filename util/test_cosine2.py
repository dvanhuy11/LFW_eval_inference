from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np 
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
# sess = tf.compat.v1.Session(config=config)
# set_session(sess)

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	print(results)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
	# extract faces
	faces = [extract_face(f) for f in filenames]
	# convert into an array of samples
	samples = asarray(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	# create a vggface model
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
	yhat = model.predict(samples)
	return yhat

def cosine_similarity(features, init_features):
    features_norm = features / np.linalg.norm(features)
    if init_features.ndim == 1:
        init_features = init_features.reshape(1, -1)
    init_features_norm = init_features / np.linalg.norm(init_features, axis=1, keepdims=True)
    similarity_scores = np.dot(features_norm, init_features_norm.T)
    return similarity_scores.squeeze()

# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
	# calculate distance between embeddings
	score = cosine_similarity(known_embedding, candidate_embedding) #neu dung cosine nay thi > threshold moi la dung 
	# score = cosine(known_embedding, candidate_embedding) #neu dung cosine nay thi < threhold moi la dung 
	if score <= thresh:
		print('>face is NOT a Match (%.3f <= %.1f)' % (score, thresh))
	else:
		print('>face is a Match (%.3f > %.1f)' % (score, thresh))
		
# define filenames
filenames = ['lfw/Aaron_Peirsol/Aaron_Peirsol_0001.jpg',
			 'lfw/Claire_Hentzen/Claire_Hentzen_0001.jpg', 
			 'lfw/Beyonce_Knowles/Beyonce_Knowles_0001.jpg', 
			 'lfw/Aaron_Peirsol/Aaron_Peirsol_0004.jpg']
# get embeddings file filenames
print(extract_face(filenames[0]))
embeddings = get_embeddings(filenames)
# define sharon stone
sharon_id = embeddings[0]

is_match(embeddings[0], embeddings[1])
is_match(embeddings[0], embeddings[2])
is_match(embeddings[0], embeddings[3])