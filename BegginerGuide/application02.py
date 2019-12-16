import tensorflow as tf
import matplotlib.pyplot as plt

from Graph import plot_images as plot

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

predictions = model.predict(test_images)

i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot.plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot.plot_value_array(i, predictions[i], test_labels)
plt.show()
