import tensorflow as tf
print(tf.__version__)

# load the tensorflow fashion_mnist module for clothes data
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# normalise the data so guessing is easier
training_images  = training_images / 255.0
test_images = test_images / 255.0

# choose the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# specify how the predictions improve
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
model.fit(training_images, training_labels, epochs=5)

# output predictions for a set of clothing pieces
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)

# print probabilities for each clothing piece for the first image
print(classifications[0])

# print the predicted clothing piece by the model
print(test_labels[0])