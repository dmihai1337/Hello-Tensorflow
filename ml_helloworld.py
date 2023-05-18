import tensorflow as tf
import numpy as np
from tensorflow import keras

# choose the model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# specify how the estimation improves
model.compile(optimizer='sgd', loss='mean_squared_error')

# define data sets
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# train model
model.fit(xs, ys, epochs=1000)

# output prediction for a value
print(model.predict([10.0]))