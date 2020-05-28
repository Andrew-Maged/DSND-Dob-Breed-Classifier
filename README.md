# Dog Breed Classification Project Using Flask

1. [Introduction](#Introduction)
2. [Project Motivation](#motivation)
3. [Files Description](#files)
4. [Steps Involved](#Steps)
5. [Instructions](#Instructions)
6. [Prerequisites](#Libraries)
7. [Licensing, Authors, and Acknowledgements](#licensing)

## Introduction: <a name="Introduction"></a>
Dog Breed classifier project of the Data Scientist Nanodegree by Udacity. A Web Application is developed using Flask through which a user can check if an uploaded image is that of a dog or human. Also, if the uploaded image is that of a human, the algorithm tells the user what dog breed the human resembles the most. The Deep Learning model distinguishes between the 133 classes of dogs with an accuracy of over 82.89%.

## Project Motivation<a name="motivation"></a>
The goal of this project is to classify images of dogs according to their breed. When the image of a human is provided, it recommends the best resembling dog breed. I decided to opt for this project as I found the topic of Deep Neural Networks to be very fascinating and wanted to dive deeper into this with some practical work.

## Description of repository:
The repository consists of the Jupyter Notebook files from the Udacity classroom, in both formats: dog_app.html and dog_app.ipynb. All credits for code examples here go to Udacity. Moreover there are files for the web application developed using Flask and contains all code necessary for running the dog breed classifier app on the local machine.

## Files Description: <a name="files"></a>
```
- haarcascades
|- haarcascade_frontalface_alt.xml #Face detection saved model
-Models
|-VGG16.h5 # a pretrained VGG16 model
- Notebooks
|- dog_app.ipynb
-static
|-bootstrap.min.css # Stylesheet file
-Templates
|-home.html
-application.py # the main app
-requirements.txt
-extract_bottleneck_features.py 
- README.md
```

## Steps Involved: <a name="Steps"></a>

1. Import Datasets
2. Detect Humans
3. Detect Dogs
4. Create a CNN to Classify Dog Breeds (from Scratch)
5. Use a CNN to Classify Dog Breeds (using Transfer Learning)
6. Create a CNN to Classify Dog Breeds (using Transfer Learning)
7. Write the Pipeline
8. Create a Flask application that enables a user to upload an image and see the results.

## Instructions: <a name="Instructions"></a>
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
1. Clone this repository `git lfs clone https://github.com/mitkir/keras-flask-image-classifier`
2. Open project's directory `cd keras-flask-image-classifier`
3. Install all necessary dependencies `pip install -r requirements.txt`
4. Run application `python application.py`
5. Open `http://127.0.0.1:5000/` on your browser
6. Click the file select button and select test image for classifier.

## Prerequisites: <a name="Libraries"></a>

1. Python 3.7+
2. Keras
3. OpenCV
4. Matplotlib
5. NumPy
6. glob
7. tqdm
8. Scikit-Learn
9. Flask
10. Tensorflow

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data) .