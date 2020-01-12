from keras.metrics import accuracy, binary_accuracy, categorical_accuracy
import tensorflow as tf
from keras import backend as K

a = [0, 0, 0, 0, 0, 1]
b = [0, 0, 1, 0, 0, 1]

with tf.Session() as sess:
    print(binary_accuracy(a, b).eval())

