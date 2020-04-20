import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import jsonify
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

labels = ['normal', 'Tüberküloz' , 'Zatürre']
labels = np.asarray(labels)
print(labels.shape)
print(labels)
num_classes = len(labels)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

	with open('model_4.json','r') as f:
    model_json = json.load(f)
    model_json = json.dumps(model_json)

def get_model(image):
		global model
		model = model_from_json(model_json)
		#model = load_model('model_4.h5')
		prediction = model.predict_classes(image)
		print("Model yüklendi!")
		print(prediction)
		print(prediction.shape)
		prediction = prediction%3
		return prediction


			
def preprocess_image(image,target_size):
		if image.mode != "RGB":
			image = image.convert("RGB")
		image = image.resize(target_size)
		image= img_to_array(image)
		image = np.expand_dims(image,axis=0)
		#image = image.reshape(-1,100,100,3)

		print(image.shape)
		
		return image


def get_commonname(idx):
		sciname = labels[idx].tolist()
		print(sciname)
		return(sciname)

	
	
@app.route("/predict", methods = ["POST"])
def predict():
		message = request.get_json(force=True)
		encoded = message['image']
		decoded = base64.b64decode(encoded)
		image= Image.open(io.BytesIO(decoded))
		processed_image = preprocess_image(image,target_size=(100,100))
		keras.backend.clear_session()
		prediction = get_model(processed_image).tolist()  # sorun burda Model NoneType olmuş amk
		sonuc = get_commonname(prediction)
		str1 = ''.join(sonuc)

		response = {
			'prediction' : str1
		}
		print(response);
		return jsonify(response)
