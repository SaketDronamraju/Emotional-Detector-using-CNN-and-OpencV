
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))




import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt

from tensorflow.keras.utils import load_img
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers import Dense,Input,Dropout,Flatten,Conv2D,Activation,MaxPooling2D,BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Reshape, SimpleRNN
from tensorflow.keras.layers import LSTM




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


classes = 7
model = Sequential()

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

# Add RNN or LSTM layer
model.add(SimpleRNN(64, return_sequences=True))  # RNN layer with return_sequences
model.add(LSTM(64))                              # LSTM layer





#1st connected layer
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
          
#2nd connected layer
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))


          
model.add(Dense(classes, activation='softmax'))
          
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


model_checkpoint = ModelCheckpoint("./model.h5", 
                                   monitor="val_acc", 
                                   verbose=1, 
                                   save_best_only=True, 
                                   mode="max")

early_stop = EarlyStopping(monitor="val_loss",
                           min_delta=0,
                           patience=3,
                           verbose=1,
                           restore_best_weights=True)

reduce_learning_rate = ReduceLROnPlateau(monitor="val_loss", 
                                         factor=0.2, 
                                         patience=3, 
                                         verbose=1,
                                         min_delta=0.0001)

callbacks_list = [early_stop, model_checkpoint, reduce_learning_rate]
epochs = 50



history = model.fit(train_set, 
                    epochs=epochs, 
                    callbacks=callbacks_list, 
                    validation_data=test_set,
                    steps_per_epoch=train_set.n//train_set.batch_size, 
                    validation_steps=test_set.n//test_set.batch_size)

model.save("./edshit.h5")


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






