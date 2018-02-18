import numpy as np
import tensorflow as tf


# pooling layers
def max_pool_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1,2,2,1], strides=[1, 2, 2, 1], padding='SAME')

def my_net(x,n_classes,keep_prob):
    # simple convnet, modified version of the one used for traffic sign classification https://github.com/rosivagyok/TrafficSignClassifier
    kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3)

    conv1_out = 32
    conv1 = tf.layers.conv2d(x, conv1_out, kernel_size=(3, 3), padding='same',
                                         activation=tf.nn.relu, kernel_regularizer=kernel_regularizer)
    pool1 = max_pool_2x2(conv1)


    conv2_out = 64
    conv2 = tf.layers.conv2d(pool1, conv2_out, kernel_size=(3, 3), padding='same',
                                         activation=tf.nn.relu, kernel_regularizer=kernel_regularizer)
    pool2 = max_pool_2x2(conv2)

    _, h, w, c = pool2.get_shape().as_list()

    # flatten our feature map (pool2) to shape 
    # [batch_size, features=[-1(dimension will be dynamically calculated based on the number of examples in our input data), h * w * c]], 
    # so that our tensor has only two dimensions:
    pool2_flat = tf.reshape(pool2, shape=[-1, h * w * c])

    # To help improve the results of the model, we also apply feedforward dropout regularization to the dense layer
    pool2_drop = tf.nn.dropout(pool2_flat, keep_prob=keep_prob)

    hidden = tf.layers.dense(pool2_drop, units=n_classes, activation=tf.nn.relu)

    # perform classification on the features extracted by the convolution/pooling layers (RAW values)
    logits = tf.layers.dense(hidden, units=n_classes, activation=None)

    predictions = tf.nn.softmax(logits)

    return predictions

