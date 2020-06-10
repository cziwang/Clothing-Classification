import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#scale data to [0,1]
train_images = train_images/255
test_images = test_images/255

#28 * 28 = 784, flatten data --> input layer in
#neural network wil have 784 nodes
#keras.sequential specifies network architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax") #Ensures that the sum of output vector is 1
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#epochs: how many times model will see the same image
#order in which images come in will influence how weights
#and biases are adjusted; epochs will repeat same images
#in different orders
#NOTE: higher epoch does not always result in better accuracy
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Accuracy:", test_acc)

prediction = model.predict([test_images[7]])

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction:" + class_names[np.argmax(prediction[i])])
    plt.show()
