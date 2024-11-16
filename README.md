# Age and Gender Detection with GUI
Project Overview
This project involves developing a machine learning model to detect the age and gender of individuals from images, integrated with a graphical user interface (GUI) for easy interaction. The project aims to provide a user-friendly tool for real-time age and gender prediction.
Table of Contents
Project Overview

Features

Installation

Usage

Dataset

Model Architecture

GUI Implementation

Acknowledgements

Features
Age Detection: Predicts the age  of the person in the image.

Gender Detection: Classifies the person in the image as male or female.

Graphical User Interface (GUI): Provides a user-friendly interface for image input and displaying results.

Installation
To get started with this project, clone the repository and install the necessary dependencies.

bash
git clone https://github.com/harimohansr/Age_Gender_detection.git
cd Age_Gender_detection
pip install -r requirements.txt
Usage
To launch the application, run the gui_1.py script. The GUI will open, allowing you to upload images and get predictions.

bash
python gui_1.py
Dataset
The model is trained on the UTKFace dataset, which includes images of faces labeled with age, gender, and ethnicity.

Dataset Name: UTKFace

Description: The dataset contains over 20,000 images with age, gender, and ethnicity labels.

Model Architecture
The model architecture consists of the following components:

Preprocessing: Converts images to grayscale, resizes them, and normalizes pixel values.

Feature Extraction: Uses convolutional neural networks (CNNs) to extract features from the images.

Classification: Two separate neural network branches for predicting age and gender.

Example Code Snippet
python
import cv2
import numpy as np
from keras.models import load_model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image

# Load pre-trained model
model = load_model('age_gender_model.h5')

# Preprocess the image
image = preprocess_image('path/to/your/imagefile.jpg')

# Predict
prediction = model.predict(image)
age, gender = prediction
print(f"Age: {age}, Gender: {gender}")
GUI Implementation
The GUI is built using [library/toolkit], providing a simple interface for users to upload images and receive predictions.

GUI Code Snippet
python
![image](https://github.com/user-attachments/assets/58cb0787-cbcb-4944-9ca7-40974ea1b0e3)

Sample output:
![image](https://github.com/user-attachments/assets/7ba60638-f244-4f5a-9d56-f2b2209c9fcd)
Uploading Image
![image](https://github.com/user-attachments/assets/328cd36f-f733-45c7-af20-3ea72e5b178b)

Detecting Image
![image](https://github.com/user-attachments/assets/5ea67d28-2004-40ae-a1aa-47889666ae74)



    
