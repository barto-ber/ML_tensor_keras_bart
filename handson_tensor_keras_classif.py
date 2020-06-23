from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt

time_start = datetime.now()

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print(X_train_full.shape)
print(X_train_full.dtype)

'''
Let's split the full training set into a validation set and a (smaller) training set. We also scale the
pixel intensities down to the 0-1 range and convert them to floats, by dividing by 255.
'''
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
# plt.show()

print("y_train:\n", y_train)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print("Label of y_train[0]:\n", class_names[y_train[0]])

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
print(model.summary())

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
'''
loss: 0.2292 - accuracy: 0.9177 - val_loss: 0.3081 - val_accuracy: 0.8904
'''
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]

# model.evaluate(X_test, y_test)
'''
loss: 72.0110 - accuracy: 0.8359
'''
X_new = X_test[:3]
y_proba = model.predict(X_new)
print("Probabilities of the 3 first instances of test data:\n",y_proba.round(2))
'''
Probabilities of the 3 first instances of test data:
 [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]
'''
y_pred = model.predict_classes(X_new)
print("Predictions with the highest estimated probabilities:\n", y_pred)
print(np.array(class_names)[y_pred])
'''
Predictions with the highest estimated probabilities:
 [9 2 1]
 ['Ankle boot' 'Pullover' 'Trouser']
'''
y_new =  y_test[:3]
print("Correct labels are:\n", y_new)
'''
Correct labels are:
 [9 2 1]
'''



