import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt

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


