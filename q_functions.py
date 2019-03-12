import tensorflow as tf
from tensorflow.python.layers.layers import *
from tensorflow.contrib.rnn import LSTMBlockFusedCell


class QFunction:
    def is_recurrent(self):
        raise NotImplementedError

    def __call__(self, state, n_actions, scope):
        raise NotImplementedError


class CartPoleNet(QFunction):
    def is_recurrent(self):
        return False

    def __call__(self, state, n_actions, scope):
        hidden = flatten(state) # flatten to make sure 2-D

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            hidden  = dense(hidden, units=512,       activation=tf.nn.tanh)
            hidden  = dense(hidden, units=512,       activation=tf.nn.tanh)
            qvalues = dense(hidden, units=n_actions, activation=None)

        return qvalues, None


class AtariRecurrentConvNet(QFunction):
    def is_recurrent(self):
        return True

    def __call__(self, state, n_actions, scope):
        state = tf.cast(state, tf.float32) / 255.0

        hidden = tf.reshape(state, [-1, state.shape[2], state.shape[3], state.shape[4]])
        print('Recurrent', state.shape)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            hidden = conv2d(hidden, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
            hidden = conv2d(hidden, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
            hidden = conv2d(hidden, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)

            batch_size = tf.shape(state)[0]
            hidden = tf.reshape(hidden, [batch_size, state.shape[1], tf.size(hidden[0])])

            hidden, new_rnn_state = self.lstm(hidden, batch_size, num_units=512)
            qvalues = dense(hidden[:, -1], units=n_actions, activation=None)

        return qvalues, new_rnn_state

    def lstm(self, inputs, batch_size, num_units, swap_axes=True):
        if swap_axes:
            inputs = tf.transpose(inputs, [1, 0, 2])
        cell = LSTMBlockFusedCell(num_units)
        self.rnn_state = self.zero_state(batch_size, num_units)
        outputs, new_rnn_state = cell(inputs=inputs, initial_state=self.rnn_state, dtype=tf.float32)
        if swap_axes:
            outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs, new_rnn_state

    def zero_state(self, batch_size, num_units):
        shape = (batch_size, num_units)
        return tuple([tf.zeros(shape), tf.zeros(shape)])


class AtariConvNet(QFunction):
    def is_recurrent(self):
        return False

    def __call__(self, state, n_actions, scope):
        state = tf.cast(state, tf.float32) / 255.0

        hidden = state
        hidden = tf.unstack(hidden, axis=1)
        hidden = tf.concat(hidden, axis=-1)
        print('Feedforward', hidden.shape)

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            hidden = conv2d(hidden, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
            hidden = conv2d(hidden, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
            hidden = conv2d(hidden, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)

            hidden = flatten(hidden)

            hidden  = dense(hidden, units=512,       activation=tf.nn.relu)
            qvalues = dense(hidden, units=n_actions, activation=None)

        return qvalues, None
