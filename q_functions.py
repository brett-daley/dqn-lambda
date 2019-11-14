import tensorflow as tf
from tensorflow.python.layers.layers import *


def cartpole_mlp(state, n_actions, scope):
    hidden = flatten(state) # flatten to make sure 2-D
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        hidden  = dense(hidden, units=512,       activation=tf.nn.tanh)
        hidden  = dense(hidden, units=512,       activation=tf.nn.tanh)
        qvalues = dense(hidden, units=n_actions, activation=None)
    return qvalues


def atari_cnn(state, n_actions, scope):
    hidden = tf.cast(state, tf.float32) / 255.0
    hidden = tf.unstack(hidden, axis=1)
    hidden = tf.concat(hidden, axis=-1)

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        hidden = conv2d(hidden, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
        hidden = conv2d(hidden, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
        hidden = conv2d(hidden, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)

        hidden = flatten(hidden)

        hidden  = dense(hidden, units=512,       activation=tf.nn.relu)
        qvalues = dense(hidden, units=n_actions, activation=None)

    return qvalues
