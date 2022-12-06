'''
Author: Aliyu Abubakar
Date: 6th December 2022
'''

import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Activation,Flatten,Dense
import keras.utils as np_utils
import argparse
import cv2
import matplotlib.pyplot as plt

'''
Get the MNIST dataset from keras datasets. It will be downloaded if you are fetching for the first time.
The data is already split and is in form of numpy array
'''
print("[INFO] Loading the MNIST dataset...")
(TRAINdata, TRAINlabels),(TESTdata,TESTlabels) = mnist.load_data()

'''
Reshape the data matrix from (samples, height, width) to (samples,height,width, depth)
'''
TRAINdata = TRAINdata[:,:,:,np.newaxis]
TESTdata = TESTdata[:,:,:,np.newaxis]

'''
Rescale the data from values between [0-255] to [0-1.0]
'''
TRAINdata = TRAINdata / 255.0
TESTdata = TESTdata / 255.0

'''
The labels come as a single digit indicating class. We need a categorical vector as the label
'''
TRAINlabels = np_utils.to_categorical(TRAINlabels, 10)
TESTlabels = np_utils.to_categorical(TESTlabels, 10)

'''
Now the core part of the code is to define the structure of the model
'''
def build_model(width, height, depth, classes):
    model = Sequential()

    #The first set of CONV=>RELU=>POOL layers
    model.add(Conv2D(16, (3,3), padding="same", input_shape=(height,width, depth)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #The second set of CONV=>RELU=>POOL layers
    model.add(Conv2D(32, (3,3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #The set of FC=>RELU layer
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))

    #The Softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    #Return the contructed network architecture
    return model

'''
Build and compile the model
'''
print("[INFO] Building and compiling the LeNet model...")
opt = SGD(lr=0.01)
model = build_model(width=28, height=28, depth=1, classes=10)
model.compile(loss = "categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

'''
Check the argument whether to train the model
'''
print("[INFO] Training the model...")
history = model.fit(TRAINdata,TRAINlabels, batch_size=4,epochs=20,validation_data=(TESTdata,TESTlabels),verbose=1)
#Use the test data to evaluate the model
print("[INFO] Evaluating the model...")
(loss, accuracy) = model.evaluate(TESTdata,TESTlabels, batch_size=4,verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

'''
Define a function to graph the training history of the model
'''
def graph_training_history(history):
    plt.rcParams["figure.figsize"] = (12,9)
    plt.style.use('ggplot')
    plt.figure(1)

    #Summarize history for accuracy
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='Lower right')

    #Summarize history for loss
    plt.subplot(212)
    plt.plot(history,history['loss'])
    plt.plot(history, history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()

    #Visualize the training history
    graph_training_history(history)

'''
Check the argument on whether to save the model weights to file
'''
print("[INFO] Saving the model weights to file...")
model.save_weights('cnn_model.h5', overwrite=True)



