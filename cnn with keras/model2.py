#note: takes quite some time


import os
import math
from tensorflow import keras
import numpy as np
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
from livelossplot import PlotLossesKerasTF
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential 
from keras._tf_keras.keras.applications import VGG16, Xception
from keras._tf_keras.keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout
from keras._tf_keras.keras.layers import MaxPooling2D, AveragePooling2D, Activation

from keras._tf_keras.keras import optimizers
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.preprocessing import image


from PIL import Image
from PIL import ImageFile

import matplotlib.image as mpimg

#GPU configs

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


# basedir = "./back/"

def removeCorruptedImages(path):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    corrupted_found = False  # Flag to track if any corrupted files are found
    
    for root, dirs, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except (IOError, SyntaxError) as e:
                print('Bad file:', file_path)
                corrupted_found = True
               

    if not corrupted_found:
        print("Good work! All files are okay.")

# Example usage
base_dir = os.path.join(os.getcwd(), 'back')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

removeCorruptedImages(train_dir)
removeCorruptedImages(validation_dir)
removeCorruptedImages(test_dir)


train_datagen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

batch_size=8
print("For Training: ")
train_datagen = train_datagen.flow_from_directory(
                  directory = train_dir,
                  target_size=(300,300),
                  batch_size=batch_size,
                  shuffle=True,
                  color_mode="rgb",
                  class_mode='categorical')

print("\nFor Testing: ")
val_datagen = test_datagen.flow_from_directory(
                directory = validation_dir,
                target_size=(300,300),
                batch_size=batch_size,
                shuffle=False,
                color_mode="rgb",
                class_mode='categorical')


base_model_path1 = "./backExerciseClassifier.h5"
# base_model = VGG16(weights=base_model_path1, include_top=False, input_shape=(300, 300, 3))
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

output_neurons = 4

model = Sequential()
model.add(Flatten(input_shape=base_model.output_shape[1:]))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(output_neurons))
model.add(Activation('softmax'))

model = Model(inputs=base_model.input, outputs=model(base_model.output))


optimizers = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
losss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2, from_logits=True)
model.compile(loss=losss,
             optimizer=optimizers,
              metrics=['accuracy'])

model.summary()

# class ConvolutionCallback(tf.keras.callbacks.Callback):
#         def on_epoch_end(self,epoch,logs={}):
#             if(logs.get('accuracy')>=0.97 and logs.get('val_accuracy') >=0.92):
#                 print("Reached greater than 97.0% accuracy so cancelling training!")
#                 self.model.stop_training = True
                
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
#                               patience=2, min_lr=0.001, mode='auto')
# # checkpoint = keras.callbacks.ModelCheckpoint("./checkpoints_models/pose_classification_model_weights2.h5", monitor='val_accuracy',
#                             #  save_weights_only=True, mode='max', verbose=1)

# convolutionCallback = ConvolutionCallback()
# # callbacks = [PlotLossesKerasTF(), checkpoint,reduce_lr, convolutionCallback]



BATCH_SIZE = 16

history = model.fit(train_datagen,
                    epochs=4,
                    validation_data = val_datagen,
                    
                    )


plt.figure(0)
plt.plot(history.history['loss'],'g', label="Loss")
plt.plot(history.history['val_loss'],'b',label="Validation Loss")
plt.plot(history.history['accuracy'],'r', label="Accuracy")
plt.plot(history.history['val_accuracy'],'black', label="Validation Accuracy")
plt.legend()
plt.show()


yoga_set1_model_save_path = "./backExerciseClassifier2.h5"

model.save(yoga_set1_model_save_path)
loaded_model = tf.keras.models.load_model('./backExerciseClassifier2.h5')
loaded_model.summary()

model.evaluate(val_datagen)


exercise_label = {0:"bridge",1:"childs pose",2:"double knee hug",3:"sphinx pose"}
# path = input("Enter Image Name (from 1-15) : ")
path = "./back/bridges4.png"


img = image.load_img(path, target_size=(300, 300))
x = image.img_to_array(img)
# print(x/255)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = loaded_model.predict(images, batch_size=10)

plt.axis("Off")
img = mpimg.imread(path)
plt.imshow(img)
plt.show()

print("Class Predictions: ",classes)
pred_index = np.argmax(classes[0])
print("\nPrediction is: ", exercise_label[pred_index])