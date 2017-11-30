import tensorflow as tf
import tensorflow.contrib.layers as layers
import logging


class DQN:
    def __init__(self, config, session, scope, obs_shape, n_actions, src_network=None):
        self.logger = logging.getLogger()
        self.session = session
        self.scope = scope
        self.obs_shape = list(obs_shape)
        self.n_actions = n_actions

        self.history_length = int(config.get('agent', 'history_length'))

        if not self.is_recurrent():
            self.obs_shape[-1] *= self.history_length

        self._create_model()

        if src_network:
            self._create_copy_op(src_network)
        else:
            self._create_train_op()

    def is_recurrent(self):
        raise NotImplementedError

    def _create_model(self):
        with tf.variable_scope(self.scope):
            self.obs_input = tf.placeholder(tf.float32, [None] + self.obs_shape, name='obs_input')
            self.tracelength = tf.placeholder_with_default(1, shape=(), name='tracelength')
            self.batch_size = tf.placeholder_with_default(1, shape=(), name='batch_size')

            self.qvalues = self._neural_network(self.obs_input, self.tracelength, self.batch_size)

            self.max_q = tf.reduce_max(self.qvalues, -1)
            self.best_action = tf.argmax(self.qvalues, -1)

    def _neural_network(self, obs_input, tracelength, batch_size):
        raise NotImplementedError

    def _create_train_op(self):
        self.target_input = tf.placeholder(tf.float32, shape=[None])
        self.action_input = tf.placeholder(tf.int32, shape=[None])

        # Get values for only the actions in this batch
        indices = tf.stack([tf.range(tf.size(self.action_input)), self.action_input], axis=1)
        self.action_values = tf.gather_nd(self.qvalues, indices)

        self.td_error = self.target_input - self.action_values
        self.loss = tf.reduce_mean(tf.square(self.td_error))

        self.train_op = self._optimization(self.loss)

    def _optimization(self, loss):
        raise NotImplementedError

    def copy(self):
        self.session.run(self.copy_op)
        self.logger.info('Copied parameters to target network: {}'.format(self.scope))

    def _create_copy_op(self, src_network):
        with tf.variable_scope(self.scope):
            copy_ops = []
            all_src_vars = [v for v in tf.trainable_variables() if v.name.startswith(src_network.scope)]

            for src_var in all_src_vars:
                # Split the src_var.name at the src_network.scope, and then replace the src_network.scope with the target scope
                target_var_name = self.scope + src_var.name.split(src_network.scope, 1)[-1]
                # Find the target var
                target_var =  [v for v in tf.global_variables() if v.name == target_var_name][0]
                self.logger.info(target_var.name)
                copy_op = target_var.assign(src_var)
                copy_ops.append(copy_op)

            self.copy_op = tf.group(*copy_ops, name='copy_op')


class AtariConvNet(DQN):
    def is_recurrent(self):
        return False

    def _neural_network(self, obs_input, _, batch_size):
        hidden = layers.convolution2d(obs_input, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
        hidden = layers.convolution2d(hidden, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
        hidden = layers.convolution2d(hidden, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)

        hidden = layers.flatten(hidden)
        hidden = layers.fully_connected(hidden, num_outputs=512, activation_fn=tf.nn.relu)

        qvalues = layers.fully_connected(hidden, num_outputs=self.n_actions, activation_fn=None)
        return qvalues

    def _optimization(self, loss):
        train_op = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.95, momentum=0.95, epsilon=0.01).minimize(loss)
        return train_op


class AtariConvRecNet(DQN):
    def is_recurrent(self):
        return True

    def _neural_network(self, obs_input, tracelength, batch_size):
        hidden = layers.convolution2d(obs_input, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
        hidden = layers.convolution2d(hidden, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
        hidden = layers.convolution2d(hidden, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)

        hidden = tf.reshape(hidden, [tracelength, batch_size, tf.size(hidden[0])])

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=512)
        self.init_state = cell.zero_state(self.batch_size, tf.float32)

        outputs, self.state = tf.nn.dynamic_rnn(cell, inputs=hidden, initial_state=self.init_state, dtype=tf.float32, time_major=True)

        qvalues = layers.fully_connected(outputs[-1], num_outputs=self.n_actions, activation_fn=None)
        return qvalues

    def _optimization(self, loss):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.95, momentum=0.95, epsilon=0.01)
        gradients = optimizer.compute_gradients(loss)
        gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]

        train_op = optimizer.apply_gradients(gradients)
        return train_op


class CartPoleNet(DQN):
    def is_recurrent(self):
        return False

    def _neural_network(self, obs_input, _, batch_size):
        hidden = layers.flatten(obs_input)
        hidden = layers.fully_connected(hidden, num_outputs=256, activation_fn=tf.nn.relu)

        qvalues = layers.fully_connected(hidden, num_outputs=self.n_actions, activation_fn=None)
        return qvalues

    def _optimization(self, loss):
        train_op = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.95, momentum=0.95, epsilon=0.01).minimize(loss)
        return train_op


class CartPoleRecNet(DQN):
    def is_recurrent(self):
        return True

    def _neural_network(self, obs_input, tracelength, batch_size):
        hidden = tf.reshape(obs_input, [tracelength, batch_size, tf.size(obs_input[0])])

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=256)
        self.init_state = cell.zero_state(self.batch_size, tf.float32)

        outputs, self.state = tf.nn.dynamic_rnn(cell, inputs=hidden, initial_state=self.init_state, dtype=tf.float32, time_major=True)

        qvalues = layers.fully_connected(outputs[-1], num_outputs=self.n_actions, activation_fn=None)
        return qvalues

    def _optimization(self, loss):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00025, decay=0.95, momentum=0.95, epsilon=0.01)
        gradients = optimizer.compute_gradients(loss)
        gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]

        train_op = optimizer.apply_gradients(gradients)
        return train_op
