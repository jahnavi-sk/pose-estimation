from keras import models
from keras._tf_keras.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the model
loaded_model = models.load_model('backExerciseClassifier.h5')

# Define the labels
exercise_label = {0: "bridge", 1: "childs pose", 2: "double knee hug", 3: "sphinx pose"}

# Path to the image for prediction
path = "./posee.png"

# Load and preprocess the image
img = image.load_img(path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.0

# Make prediction
images = np.vstack([x])
classes = loaded_model.predict(images, batch_size=10)

# Display the image
plt.axis("Off")
img = mpimg.imread(path)
plt.imshow(img)
plt.show()

# Print prediction results
print("Class Predictions: ", classes)
pred_index = np.argmax(classes[0])
print("\nPrediction is: ", exercise_label[pred_index])
