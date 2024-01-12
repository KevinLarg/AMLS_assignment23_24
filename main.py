# ALMS 1 ELEC0134
# Term 1 Final Assignment
# By SN 23043422, from 13/12/2023 to ??/01/2024

# This file aim to address all main code parts used when solving task A and B described in the task sheet;

# Environment: Python 3.12.0

# Library installed (as the order to be used): medmnist, numpy,

# Datasets for both tasks, for task A and for task B respectively, are installed by
# %pip install --upgrade git+https://github.com/MedMNIST/MedMNIST.git

# Task A

#importing libraries
import medmnist
import numpy as np
import pandas as pd
#print("successfully installed madmnist, version:", medmnist.__version__)
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import tensorflow as tf
import tensorflow.keras as Keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

import warnings
import joblib
from sklearn.decomposition import PCA

def load_data(datapath):
    """
    importing training, validating and testing data seperately
    """
    try:
        data = np.load(datapath)
    except:
        print("loading data failed, please check the file path provided")
    data_train = data['train_images']
    data_test = data['test_images']
    data_val = data['val_images']
    label_train = data['train_labels']
    label_test = data['test_labels']
    label_val = data['val_labels']
    return data_train,label_train,data_val,label_val,data_test,label_test

def flatten_data(data):
    """
    flattening the data for some ml models
    (task A)
    """
    n_samples = len(data)
    data_temp = data.reshape((n_samples, -1))
    data_tr = StandardScaler().fit_transform(data_temp)
    return data_tr

def reshape_data(data):
    """
    reshaping data so that it gets the forth dimension 
    i.e. number of channels
    """
    inshape = (28, 28, 1)  # image length, width and channels, which is 28,28,1 in task A
    train_reshaped = np.reshape(data, (data.shape[0],) + inshape)
    return train_reshaped

def SVM_models(kernels, C, data_tr,label_train,data_va,label_val,data_te,label_test):
    # Initialize an empty dictionary to store results
    accuracy_results = {'kernel': [], 'C': [], 'accu': [], 'valscore': []}

    for i in kernels:
        for c in C:
            # Create an SVM classifier
            clf_svm = SVC(kernel= i , C = c)

            # Train the classifier
            clf_svm.fit(data_tr, label_train)

            # Make predictions on the test set
            y_pred = clf_svm.predict(data_te)

            val_scores = cross_val_score(clf_svm, data_va, label_val, cv=5)

            # Evaluate the performance
            accu_svm = accuracy_score(label_test, y_pred)

            # storing performance values for plotting
            accuracy_results['kernel'].append(i)
            accuracy_results['C'].append(c)
            accuracy_results['accu'].append(accu_svm)
            accuracy_results['valscore'].append(val_scores.mean())
            print(f"for SVM mdoel with a {i} kernel and c = {c},\n validaction acc is {val_scores.mean()}\n Test acc is {accu_svm}")
    return accuracy_results

def load_trained_model(datapath):
    try:
        model_reload = load_model(filepath= datapath)
        print('Loading model; Success')
    except:
        print('''Loading model failed; This is probably due to an OS error encountered at the last checking stage:
              Cannot parse keras metadata at path A/my_saved_model\keras_metadata.pb: Received error: Error parsing message
              ''')
    return model_reload

def cnn_evaluate(CNN_A,test_reshaped, label_test):
    """
    showing cnn test accuracy and confusion matrix
    """
    result_cnn = np.round(CNN_A.predict(test_reshaped),0)
    print("Convolutional Neural Network CNN results\nConfusion Matrix:")
    print(confusion_matrix(label_test, result_cnn))
    test_loss, test_accuracy = CNN_A.evaluate(test_reshaped, label_test)
    print(f"test accuracy is now {test_accuracy} with a loss of {test_loss}")
    return 

def rf_model(data_tr,label_train,data_va,label_val,data_te,label_test,random_seed):
    # Create a Random Forest model
    rf_modela = RandomForestClassifier(n_estimators=100, random_state=random_seed)

    # Train the model
    rf_modela.fit(data_tr, label_train)

    # Perform cross-validation
    cv_scores = cross_val_score(rf_modela, data_va, label_val, cv=5)  
    print("Random Foresr RF results:\nMean Cross-Validation Score:", cv_scores.mean())

    # Make predictions on the test set
    y_pred = rf_modela.predict(data_te)

    # Evaluate the model
    accu_rf = accuracy_score(label_test, y_pred)
    print("Test Accuracy:", accu_rf)

    # Display classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(label_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(label_test, y_pred))

    return

def load_SVM(model_path,data_te,label_test):
    """
    load the saved SVM model
    """
    print("loading SVM for task B might take up to 10 mins")
    loaded_model = joblib.load(model_path)

    y_pred = loaded_model.predict(data_te)

    accu_svm = accuracy_score(label_test, y_pred)
    print("Task B\nSVM:")
    print(classification_report(label_test, y_pred))
    return

def SVM_choose(data_tr,label_train,data_va,label_val,data_te,label_test):
    """
    Task B: please choose if 
    0: results from testcode_B directly pasted and printed
    1: train a new SVM with PCA """
    choice = '0'
    choice = input("Training a SVM for task B may take up to 25 mins.\n Please specify if you'd like to (1) train a new one or (0) see the printed result\n")
    if choice == '0':
        print("""Report for SVM with rbf kernal and regularization parameter = 1:
              precision    recall  f1-score   support

           0       0.94      0.91      0.93      1338
           1       0.89      1.00      0.94       847
           2       0.38      0.70      0.49       339
           3       0.58      0.29      0.39       634
           4       0.94      0.64      0.76      1035
           5       0.48      0.50      0.49       592
           6       0.33      0.19      0.24       741
           7       0.66      0.40      0.50       421
           8       0.55      0.88      0.68      1233

    accuracy                           0.67      7180
   macro avg       0.64      0.61      0.60      7180
weighted avg       0.69      0.67      0.66      7180
""")
    else:
        kernels = {'rbf'}
        C = {100}
        pca_used = PCA(n_components= 500) # more than 95% achieved by 500 PCs
        data_tr_pca = pca_used.fit_transform(data_tr)
        data_va_pca = pca_used.transform(data_va)
        data_te_pca = pca_used.transform(data_te)
        print("Training a rbf kernel SVM with c = 100, after PCA")
        results = SVM_models(kernels, C, data_tr_pca,label_train,data_va_pca,label_val,data_te_pca,label_test)

    return 






# To ignore all warnings
warnings.filterwarnings("ignore")
random_seed = 42
global res_size
res_size = (224,224)

# Task A
# Loading data 
data_train,label_train,data_val,label_val,data_test,label_test = load_data('Dataset/pneumoniamnist.npz')

# Preprocessing data
data_tr = flatten_data(data_train)
data_va = flatten_data(data_val)
data_te = flatten_data(data_test)

train_reshaped = reshape_data(data_train)
val_reshaped = reshape_data(data_val)
test_reshaped = reshape_data(data_test)

# showing SVM models and results
# Tried kernals and regularization parameters
kernels = ['linear', 'rbf', 'poly','sigmoid']
C = [0.1,1,10,100,1000,10000]

#accuracy_results = SVM_models(kernels, C, data_tr,label_train,data_va,label_val,data_te,label_test)
#print(f'Task A SVM results: {accuracy_results}')

# Loading CNN model and show results
cnn_reload = load_trained_model('A/my_saved_model')

cnn_evaluate(cnn_reload, test_reshaped, label_test)

# showing RF models and results
rf_model(data_tr,label_train,data_va,label_val,data_te,label_test, random_seed= random_seed)







# Task B
# Loading data 
data_train,label_train,data_val,label_val,data_test,label_test = load_data('Dataset/pathmnist.npz')

# Preprocessing data
data_tr = flatten_data(data_train)
data_va = flatten_data(data_val)
data_te = flatten_data(data_test)

data_train_res = Keras.applications.resnet50.preprocess_input(data_train)
data_valid_res = Keras.applications.resnet50.preprocess_input(data_val)
data_test_res = Keras.applications.resnet50.preprocess_input(data_test)

label_train_one_hot = to_categorical(label_train, num_classes=9)
label_valid_one_hot = to_categorical(label_val, num_classes=9)
label_test_one_hot = to_categorical(label_test, num_classes=9)

# showing SVM models and results
#load_SVM("B/svm.model", data_te, label_test)
SVM_choose(data_tr,label_train,data_va,label_val,data_te,label_test)

# showing Resnet-50 models and results
print("Task B: Loading Resnet-50 model")
resnet = load_trained_model('B/my_saved_model')
test_loss, test_accuracy = resnet.evaluate(data_test_res, label_test_one_hot)
print(f"Resnet-50: test accuracy is now {test_accuracy}")
print("Task B: Resnet-50 model with the padded data achieved a test acc of 0.7798050045967102")

# For saving time and repository space, the RF models for task B will be directly printed
print("Task B: Random Forest model with mini-batches achieved a test acc of 0.5906685236768803")
print("Task B: Random Forest model with the whole training data involved achieved a test acc of 0.6497214484679665")