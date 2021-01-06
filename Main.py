#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np
import tensorflow as tf
from mnist import MNIST
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import StratifiedKFold
# from tensorflow import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
#import tensorflow_datasets as tfds

def load_data():
    """
    loads data from local folder Data
    converts the array.array objects to numpy ndarrays
    """
    mndata = MNIST('./Data/MNIST_Data')
    train_X, train_Y = mndata.load_training()
    test_X, test_Y = mndata.load_testing()
    train_X =np.asarray(train_X)
    test_X = np.asarray(test_X)
    train_Y = np.asarray(train_Y)
    test_Y = np.asarray(test_Y)
    return train_X, train_Y, test_X, test_Y



def load_my_data():
    
    """
    loads data from local folder emilysdata
    """
    myLabels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    myImages = np.zeros((10, 28, 28, 1), dtype=np.float32)
    folderpath= "Data/emily_data/numbers"
    filename = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    for i in range(10):
        color_img = cv2.imread("%s/%s.png" % (folderpath, filename[i]))
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        # name = "test_binary" + str(i) + ".png"
        ret,binarized_img = cv2.threshold(gray_img,100,255,cv2.THRESH_BINARY_INV)
        binarized_img = np.reshape(binarized_img, (28, 28, 1))
        # cv2.imwrite(name, binarized_img)
        myImages[i, :, :, :] = binarized_img
    
    return myImages, myLabels  



# In[ ]:


def preprocess_image_data(image_data):
    """
    takes in an array object that is assumed to be X image data for MNIST.
    reshape to get grayscale 28x28 images for each row.
    converts array to float and normalizes values betweeen 0 and 1
    
    param image_data: a array.array object that is training or test data
    return image_array_norm: normalized image array
    """
    image_array = np.reshape(image_data, (image_data.shape[0], 28, 28, 1))
    image_array = image_array.astype(np.float32)
    image_array_norm = image_array / 255.0
    return image_array_norm


# In[ ]:


def preprocess_label_data(label_data):
    """
    takes in an array object and reshapes to 2D array. 
    One hot encodes labels since they are categorical.
    
    param label_data: label data
    return encoded_labels: (-1,10) array of encoded data labels
    """
    label_array = label_data.reshape(-1, 1)
    hot_encoder = OneHotEncoder(dtype=np.uint8)
    hot_encoder.fit(label_array)
    encoded_labels = hot_encoder.transform(label_array).toarray()
    return encoded_labels


# In[ ]:


# find out if dataset is balanced 
def visualize_balance_of_dataset(y, name):
    """
    output bar chart showing number of elements
    for multiclass (0, 1, 2,...9). 
    
    Used to visualize how balanced the data set is. 
    
    param y: label array
    """
    u, counts = np.unique(y, return_counts=True)
    sum_counts = np.sum(counts)
    distro_list = []
    for i in counts:
        distro =(i / sum_counts) * 100
        distro_list.append(distro)
    # print('distribution = ', distro_list)    
    plt.figure(figsize=(10, 5))
    if name == "Train":
        col = "blue"
    else:
        col = "red"
    plt.bar(u, counts, color=col)
    plt.title(name + " Dataset Distribution")
    plt.xticks(np.arange(min(u), max(u)+1, 1.0))
    plt.xlabel("Label Categories - Numerical Characters")
    plt.ylabel("Number of Label Category Occurrences")
    plt.savefig(name + "_barChart.png")


# In[ ]:


def create_CNNmodel():
    
    """
    Creates the CNN model 
    
    return model
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# In[ ]:


def cross_validate_model(Xtrain, Ytrain):
    """
    A function to perform KFold cross validation on the data
    
    param Xtrain: the training image data
    param Ytrain: the labels for the training image data
    
    return history_list, accuracy_list : loss and accuracy for each Kfold iteration
    """
    history_list = []
    accuracy_list = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=10)
    for i, j in kfold.split(Xtrain):
        Xtrain_fold, Ytrain_fold = Xtrain[i], Ytrain[i]
        XVal_fold, YVal_fold = Xtrain[j], Ytrain[j]
        
        cnn = create_CNNmodel()
        history = cnn.fit(Xtrain_fold, Ytrain_fold, epochs=10, batch_size=32, validation_data=(XVal_fold, YVal_fold), verbose=0)
        _, acc = cnn.evaluate(XVal_fold, YVal_fold, verbose=1)
        
        history_list.append(history)
        accuracy_list.append(acc)
        print('accuracy = ', (acc * 100))
    return history_list, accuracy_list


# In[ ]:


def train_evaluate(Xtrain, Ytrain, Xtest, Ytest):
    """
    trains the model with the entire training set (assumes that cross validation produced good results).
    Evaluates model on Test data provided by MNIST
    
    param: Xtrain, Ytrain, Xtest, Ytest: training and test data
    return history, results: loss and accuracy 
    """
    cnn = create_CNNmodel()
    history = cnn.fit(Xtrain, Ytrain, batch_size=32, epochs=15, validation_split=0.1)
    results = cnn.evaluate(Xtest, Ytest, verbose=1)
    cnn.save('cnn_model_new')
    return history, results



def run_from_saved_model_with_Test_Data():
    """
    loads pre-trained model and evaluates the model using the MNIST test data
    
    return results: accuracy of model on test data
    """
    X_train_i, Y_train_i, X_test_i, Y_test_i = load_data()
    # preprocess training and test labels 
    Xtest = preprocess_image_data(X_test_i)
    Ytest = preprocess_label_data(Y_test_i)
    # load pre-trained model
    cnn = load_model('cnn_model')
    results = cnn.evaluate(Xtest, Ytest, verbose=1)
    return results


# In[ ]:


def run_from_beginning():
    """
    option to train model from scratch and then evaluates using MNIST test data
    """
    X_train_i, Y_train_i, X_test_i, Y_test_i = load_data()
    # preprocess training and test labels 
    Xtrain = preprocess_image_data(X_train_i)
    Xtest = preprocess_image_data(X_test_i)
    Ytrain = preprocess_label_data(Y_train_i)
    Ytest = preprocess_label_data(Y_test_i)
    
    # train and evaluate model using training and test data
    history, results = train_evaluate(Xtrain, Ytrain, Xtest, Ytest)
    
    return history, results
   


# In[ ]:


def run_from_saved_model_with_my_data():
    """
    loads pre-trained model and evaluates the model in my personally created handwritten
    number dataset
    """
    x, y = load_my_data()
    # preprocess training and test labels 
    X_data = preprocess_image_data(x)
    Y_data = preprocess_label_data(y)
    
    # load pre-trained model
    cnn = load_model('cnn_model')
    results = cnn.evaluate(X_data, Y_data, verbose=1)
    predictions = cnn.predict(X_data)
    print('results', results)
    # print('predictions', predictions)
    return results



def run_cross_validation():
    """
    loads pre-trained model and evaluates the model using the MNIST test data
    
    return results: accuracy of model on test data
    """
    X_train_i, Y_train_i, X_test_i, Y_test_i = load_data()
    Xtrain = preprocess_image_data(X_train_i)
    Ytrain = preprocess_label_data(Y_train_i)
    cross_validate_model(Xtrain, Ytrain)



def main():
    # print command line arguments
    
    for arg in sys.argv[1:]:
        if arg == "1":
            # print("running pre-trained model on Test Data")
            print()
            sys.stdout.write("running pre-trained model on Test Data")
            print()
            result = run_from_saved_model_with_Test_Data()
            print('accuracy = ', result[1])
        if arg == "2":
            print()
            print("running pre-trained model on the data I created")
            print()
            results = run_from_saved_model_with_my_data()
            print()
            print("accuracy from saved model and my Data = ", results[1])
            print()
        if arg == "3":
            print()
            print("training  model and testing model with Test Data")
            h, r = run_from_beginning()
            print()
            print('history = ', h)
            print()
            print('results = ', r)
        if arg == "4":
            print()
            print("running cross validation on training data")
            print()
            run_cross_validation()
        if arg == "5":
            print()
            print('creating and saving bar chart that shows distribution of categories')
            X_train_i, Y_train_i, X_test_i, Y_test_i = load_data()
            visualize_balance_of_dataset(Y_train_i, "Train")
            visualize_balance_of_dataset(Y_test_i, "Test")

if __name__ == "__main__":
    main()
