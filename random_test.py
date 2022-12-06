
import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Activation,Flatten,Dense
import keras.utils as np_utils
from keras.models import load_model
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
TESTdata = TESTdata[:,:,:,np.newaxis]
'''
Rescale the data from values between [0-255] to [0-1.0]
'''
TESTdata = TESTdata / 255.0
TESTlabels = np_utils.to_categorical(TESTlabels, 10)



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
    model.add_weight('cnn_model.h5')

    #Return the contructed network architecture
    return model

'''
Build and compile the model
'''
print("[INFO] Building and compiling the LeNet model...")
opt = SGD(lr=0.01)
model = build_model(width=28, height=28, depth=1, classes=10)
model.compile(loss = "categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


#Load the weights
#model = load_model('cnn_model.h5')

'''
Select vew samples randomly from the test dataset to evaluate the model
'''
for i in np.random.choice(np.arange(0, len(TESTlabels)), size=(10,)):
    probs = model.predict(TESTdata[np.newaxis,i])
    prediction = probs.argmax(axis=1)

    #convert the digit data to a color image
    image = (TESTdata[i] * 255)#.astype("unit32")
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    #The image are in 28x28 size. Much too small to see properly
    #So, let resize them 280x280 for viewing
    image = cv2.resize(image, (280,280), interpolation=cv2.INTER_LINEAR)

    #Add the predicted value on to the image
    cv2.putText(image, str(prediction[0]),(20,20), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,0),1)

    #Show the image and prediction
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0], np.argmax(TESTlabels[i])))
    plt.imshow(image)
    plt.show()
    cv2.waitKey(10)

#Close all OpenCV windows
cv2.destroyAllWindows()


