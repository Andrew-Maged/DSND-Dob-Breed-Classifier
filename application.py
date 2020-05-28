from flask import *  
from sklearn.datasets import load_files       
from keras.utils import np_utils
from keras.models import load_model
import numpy as np
from glob import glob
import cv2                
import matplotlib.pyplot as plt 
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image                  
from tqdm import tqdm
from PIL import ImageFile 
import os
from random import randint
from extract_bottleneck_features import *

from PIL import Image                    
import json


UPLOAD_FOLDER = 'uploads'
VGG16_model = load_model('Models/vgg16.h5')

dog_names=[]
with open('Data/dog_names.json') as json_file:
    dog_names = json.load(json_file)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

	
# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
	

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))
	
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
	
	
def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
	
from PIL import Image

def which_breed(img_path):
    ## handle cases for a human face, dog, and neither
    img = Image.open(img_path)
    plt.imshow(img)
    plt.show()
    predicted_breed = VGG16_predict_breed(img_path)
    if dog_detector(img_path):
        return "Hello, Doggie!! It looks like a " + str(predicted_breed).split(".",1)[1] 
    if face_detector(img_path):
        return "Hello, Human! If you were a dog..You may look like a " + str(predicted_breed).split(".",1)[1] 
        
    else:
        return "Confused! Can't decide either dog or human.."


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        output = which_breed(file_path)
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)
