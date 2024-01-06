# Emotional-Detector-using-CNN-and-OpencV
This is an Emotional detector that was made using a CNN model which is being implemented on a live camera using the OpenCV modules

#Emotion Detector

1.CNN Model file 
We build the CNN model using tensorflow and keras and use other python modules to plot the loss and accuracy results 
We use a Kaggle notebook to build this as we can directly load the dataset without having to download the dataset
The dataset used for this model is Face expression recognition dataset by JONATHAN OHEIX
link for the dataset-https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset

Code:

import numpy as np  
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

importing the required deeplearning modules
from tensorflow.keras.utils import load_img
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers import Dense,Input,Dropout,Flatten,Conv2D,Activation,MaxPooling2D,BatchNormalization 
from keras.models import Model, Sequential 
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Reshape, SimpleRNN
from tensorflow.keras.layers import LSTM

Setting up the training set to be used and viewing the dataset
import random
expressions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
random.shuffle(expressions)
plt.figure(figsize=(12, 12))
index = 1

for i in range(1,37):
    expression = random.choice(expressions)
    image_path = os.path.join(path, 'train', expression)
    image_files = os.listdir(image_path)
    random.shuffle(image_files)
    image = load_img(os.path.join(image_path, image_files[i]), target_size=(48, 48))
    plt.subplot(6, 6, index)
    plt.imshow(image)
    plt.title(expression)
    plt.axis('off')
    index += 1

plt.tight_layout()
plt.show()

Splitting the dataset into training and validation i.e test sets
batch_size = 128
train_datagenerator = ImageDataGenerator()
validation_datagenerator = ImageDataGenerator()

train_set = train_datagenerator.flow_from_directory(path+'train',
                                                   target_size=(48, 48), 
                                                   color_mode="grayscale",
                                                   class_mode="categorical", 
                                                   batch_size=batch_size, 
                                                   shuffle=True)
test_set = validation_datagenerator.flow_from_directory(path+'validation',
                                                       target_size=(48, 48), 
                                                       color_mode="grayscale",
                                                       class_mode="categorical", 
                                                       batch_size=batch_size, 
                                                       shuffle=False)


#making the CNN layers
classes = 7
model = Sequential()
#The Sequential model is a linear stack of layers, where you can simply add one layer at a time. It is a convenient way to build deep learning models for simple scenarios where the data flows sequentially through each layer of the model.


#
Batch Normalization (BatchNorm or BN) is a technique used in deep neural networks to improve the training stability and convergence speed. It normalizes the input of each layer in a mini-batch by subtracting the batch mean and dividing by the batch standard deviation. Batch Normalization is typically applied before the activation function of a neural network layer.


#Activation lets us decide the acitvation function we want to use for the nodes
#MaxPooling2D helps us build the pool layer
#COnvo2D helps us build the convoluted layer 
#Dropout is a regularization technique used in neural networks to prevent overfitting. Overfitting occurs when a model learns the training data too well, including noise and random fluctuations, and as a result, it performs poorly on new, unseen data. Dropout helps mitigate overfitting by randomly setting a fraction of input units (neurons) to zero during each update of the training phase.

#1st CNN layer
model.add(Conv2D(64, (3,3), padding="same", input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#2nd CNN layer
model.add(Conv2D(128, (5,5), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#3rd CNN layer
model.add(Conv2D(512, (3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#4th CNN layer
model.add(Conv2D(512, (3,3), padding="same"))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Print the output shape
print(model.output_shape)  # Add this line in your code

# Reshape for RNN or LSTM
model.add(Reshape((9, 512)))  # Adjust based on the actual output shape

Add RNN or LSTM layer
 In an RNN, the network maintains hidden states that capture information about previous inputs in the sequence. This hidden state is updated at each time step and influences the processing of the current input.


However, standard RNNs have limitations in capturing long-term dependencies, known as the vanishing gradient problem, which makes them less effective for tasks that require modeling long-range dependencies in the input sequence.
#LSTMs are a specific type of RNN designed to overcome the vanishing gradient problem and better capture long-term dependencies.
LSTMs have a more complex architecture with a memory cell, input gate, forget gate, and output gate. These gates allow LSTMs to selectively read, write, and forget information in the memory cell, enabling them to maintain information over longer sequences.
The use of these gates helps LSTMs mitigate the vanishing gradient problem, making them more effective for tasks that require modeling dependencies over extended time periods.


model.add(SimpleRNN(64, return_sequences=True))  # RNN layer with return_sequences
model.add(LSTM(64))                              # LSTM layer





#fully connected 1st layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
          
#fully connected 2nd layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

#The Dense layer is a fully connected layer in a neural network. Each neuron in a dense layer is connected to every neuron in the previous layer.
#The dense layer performs a linear transformation on its input, followed by an activation function. The linear transformation involves multiplying the input by a weight matrix and adding a bias term.

#The softmax activation function is commonly used in the output layer for multi-class classification problems.
It takes a vector of raw scores (logits) as input and converts them into a probability distribution over multiple classes.
The output of the softmax function is a probability distribution, where each element represents the probability of the corresponding class.
The softmax function is  (z)i = euler's number e^zi/sigma j=1 to N e^zj   
      
model.add(Dense(classes, activation='softmax'))

#Adam Optimizer dynamically adjusts the learning rates for each parameter. It maintains separate learning rates for each parameter based on the past gradients, which helps handle different scales and variations in the data.
The adaptive learning rate mechanism allows Adam to converge faster and handle sparse data.  
Adam includes a momentum term that helps accelerate convergence, especially in the presence of noisy or sparse gradients.
The momentum term enables the optimizer to continue moving in the previously determined direction with a certain inertia, improving convergence on flat or curved loss surfaces.

#One-Hot Encoding:
The ground truth labels are often represented using one-hot encoding. In this representation, a class label is represented as a vector with all zeros except for the index corresponding to the true class, which is set to 1.
For example, if there are three classes (A, B, C), the one-hot encoding for class B might be [0, 1, 0].

#loss function 
Categorical Crossentropy computes the cross-entropy loss between the predicted probability distribution and the true distribution.
The formula for categorical crossentropy between two probability distributions 
p and q is given by: -sigma x Px.log(qx)
p is true probability(ground truth) and q is predicted probability for class x

  

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()



#model.h5' is the file name where the weights will be saved.
monitor='val_accuracy' specifies the metric to monitor for saving the best weights. The 'val_accuracy' indicates the validation accuracy.
save_best_only=True ensures that only the best weights (according to the monitored metric) are saved.
mode='max' means that the checkpoint will be saved when the monitored metric is at its maximum. Other options include 'min' and 'auto' depending on whether you are monitoring a metric you want to maximize or minimize.
verbose=1 means that the callback will provide output messages about the saving process.

model_checkpoint = ModelCheckpoint("./model.h5", 
                                   monitor="val_acc", 
                                   verbose=1, 
                                   save_best_only=True, 
                                   mode="max")
#chooses the best parameters and stops epcohs when met with best parameter values
early_stop = EarlyStopping(monitor="val_loss",
                           min_delta=0,
                           patience=3,
                           verbose=1,
                           restore_best_weights=True)
                           
                           
#
ReduceLROnPlateau is a callback in Keras that dynamically adjusts the learning rate during training based on certain conditions. The purpose is to improve the convergence and performance of the model by reducing the learning rate when the training has plateaued, i.e., when there is little or no improvement in the model's performance on a chosen metric.
monitor:
The metric to be monitored for learning rate reduction. For example, 'val_loss' or 'val_accuracy'.
factor:
The factor by which the learning rate will be reduced. For example, if factor=0.5, the learning rate will be reduced to half when the specified condition is met.
patience:
Number of epochs with no improvement on the monitored metric after which the learning rate will be reduced.
mode:
Specifies whether to reduce the learning rate when the monitored quantity has stopped increasing ('min'), stopped decreasing ('max'), or just when there is no improvement ('auto').
min_delta:
Minimum change in the monitored quantity to qualify as an improvement.
cooldown:
Number of epochs to wait before resuming normal operation after learning rate has been reduced.
min_lr:
A lower bound on the learning rate. The learning rate will not be reduced beyond this threshold.

reduce_learning_rate = ReduceLROnPlateau(monitor="val_loss", 
                                         factor=0.2, 
                                         patience=3, 
                                         verbose=1,
                                         min_delta=0.0001)

callbacks_list = [early_stop, model_checkpoint, reduce_learning_rate]
epochs = 50
#run more epochs as it improves the accuracy and reduces losses more epochs however takes alot of time to train and run,ideal 50 epochs


#history-This object contains information about the training process, including the loss and metric values recorded during each epoch
history = model.fit(train_set, 
                    epochs=epochs, 
                    callbacks=callbacks_list, 
                    validation_data=test_set,
                    steps_per_epoch=train_set.n//train_set.batch_size, 
                    validation_steps=test_set.n//test_set.batch_size)

model.save("./model.h5")#saves the model required


#plotting the loss and accuracy of the model
plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.ylabel('Loss', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')
plt.legend(loc='upper right')

plt.subplot(1,2,2)
plt.ylabel('Accuracy', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Testing Accuracy')
plt.legend(loc='lower right')
plt.show()


2.Run file
We can use any ide like pycharm,spyder,vscode,atom or even jupyter notebooks 
#we might face issues with the tesnorflow and keras calls though in this case the code was excetued on spyder 
after we run the code we can exit the program by clicking on q

Code:
#importing the modules to call our model and use the live camera
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np


#Here, a CascadeClassifier object is created using the Haar Cascade classifier for face detection. The haarcascade_frontalface_default.xml file is a pre-trained classifier specifically designed for detecting frontal faces. point to the location on the local machine where this XML file is stored. Make sure to replace this path with the actual path on your system.
This classifier is part of the Haar Cascade classifiers provided by OpenCV, and it is trained to recognize patterns that resemble faces. Haar Cascade classifiers use a machine learning approach to identify objects or features in images, and they are particularly efficient for real-time object detection.

face_classifier = cv2.CascadeClassifier(r'C:\Users\Saket\OneDrive\Desktop\Emotion detector\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\Saket\OneDrive\Desktop\Emotion detector\edshit.h5')#loads the CNN model built earlier


emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

#using opencv we use the VideoCapture function to use the live camera
cap = cv2.VideoCapture(0)


#we first convert the images to grayscale as it helps in easier computing 
while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

#creates rectanglur shape over the face
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)#makes the rectangle green,format is in BGR
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


#converts the image data into an array
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

#this function gives the label which has the maximum value closest to 1 and then displays the label in green
            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)#green color text
#if the face is not recognisable then it shows no faces detected          
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)#green color text
    cv2.imshow('Emotion Detector',frame)
#byb pressing q we can terminate the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


