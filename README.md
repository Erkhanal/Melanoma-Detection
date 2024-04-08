# Melanoma Detection

## Table of Contents:
* [Introduction](#introduction)
* [Problem Statement](#problem-statement)
* [Objectives](#objectives)
* [Analysis Approach](#analysis-approach) 
* [Technologies Used](#technologies-used)
* [References](#references)
* [Contact Information](#contact-information)

## Introduction:  
<div align="justify">This repository contains the code to build a convolutional neural network (CNN) model to accurately detect melanoma, a deadly type of cancer, from skin images. The model is trained on a dataset containing images of various oncological diseases, including melanoma, acquired from the International Skin Imaging Collaboration (ISIC). The goal is to develop a robust classification model that can assist dermatologists to detect melanoma accurately. </div>

## Problem Statement:  
<div align="justify">
We have a dataset, which has 2357 images showing cancerous and non-cancerous skin diseases. These images were collected by the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant. 

The data set contains the following diseases:

- Actinic keratosis
- Basal cell carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented benign keratosis
- Seborrheic keratosis
- Squamous cell carcinoma
- Vascular lesion

We need to make a CNN based model which can accurately detect melanoma. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis. </div>


## Objectives:  
<div align="justify">The main objective is to build a multiclass classification model using a custom convolutional neural network in TensorFlow, that can assist dermatologists to detect melanoma accurately and efficiently. </div>

## Analysis Approach:    
<div align="justify">To tackle this problem effectively, I have established a structured data analysis approach. <br>

- Data Reading/Data Understanding:<br> It includes understanding, defining the path for train and test images and exploring the dataset to understand its structure and contents.

- Dataset Creation:<br>It includes splitting the train directory into training and validation sets with a batch size of 32 and resizing all images to 180x180 pixels to maintain uniformity.

- Dataset Visualization:<br>Visualizing instances of all nine classes present in the dataset to gain insights into the data.

- Model Building & Training:<br>It includes building a custom CNN model to detect the nine classes, rescaling images to normalize pixel, choosing an appropriate optimizer and loss function, training the model for 20 epochs, evaluating model performance and analyze for signs of overfitting or underfitting.

- Data Augmentation:<br>It includes Implementing a data augmentation strategy to address overfitting or underfitting issues.

- Model Building & Training on Augmented Data:<br>It includes rebuilding a CNN model, choosing an appropriate optimizer and loss function, training the model for 20 epochs, evaluating model performance and analyzing for signs of overfitting or underfitting.

- Class Distribution Analysis:<br>It includes examining the current class distribution in the training dataset, Identify the class with the least number of samples, determining which classes dominate the data in terms of proportionate number of samples. 

- Handling Class Imbalances:<br>It includes rectifying class imbalances present in the training dataset using the Augmentor library.

- Model Building & Training on Rectified Class Imbalance Data:<br>It includes rebuilding the CNN model with rectified class imbalance data, choosing the same optimizer and loss function for training, training the model for 30 epochs, evaluating if earlier issues have been resolved and summarizing the performance of the model. </div>

## Technologies Used:
- Python, version 3 
- NumPy for numerical computations
- Matplotlib and seaborn for data visualization
- Pandas for data manipulation
- Statsmodels for statistical modeling
- Sklearn for machine learning tasks
- Tensorflow, keras, augmentor for deep learning 
- Jupyter Notebook for interactive analysis

## References:
- Python documentations
- Tensorflow, keras, augmentor documentations
- Stack Overflow


## Contact Information:
Created by https://github.com/Erkhanal - feel free to contact!
