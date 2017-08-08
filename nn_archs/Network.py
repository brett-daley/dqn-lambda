import tensorflow as tf


class Network(object):
	def __init__(self, cfg_parser, sess, scope, trainable):
		self.cfg_parser = cfg_parser
		self.sess = sess
		self.copy_op = None
		self.trainable = trainable
		self.scope = scope
		self.var = {}

	def run_copy(self):
		if self.copy_op is None:
			raise Exception('Run `create_copy_op` first before copy')
		else:
			self.sess.run(self.copy_op)
			print 'Prediction -> Target NN copy successful for', self.scope

	def create_copy_op(self, src_network):
		with tf.variable_scope(self.scope):
			copy_ops = []
			all_src_vars = [v for v in tf.trainable_variables() if v.name.startswith(src_network.scope)]

			for src_var in all_src_vars:
				# Split the src_var.name at the src_network.scope, and then replace the src_network.scope with the target scope
				target_var_name = self.scope + src_var.name.split(src_network.scope, 1)[-1]
				# Find the target var
				target_var =  [v for v in tf.global_variables() if v.name == target_var_name][0]
				print target_var.name
				copy_op = target_var.assign(src_var)
				copy_ops.append(copy_op)

			print '--------'
			self.copy_op = tf.group(*copy_ops, name='copy_op')
