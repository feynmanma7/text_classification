import tensorflow as tf
tf.random.set_seed(7)
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten


class PretrainedTextCNN(tf.keras.Model):
	def __init__(self,
				 filters=10,
				 kernel_size=5,
				 dense_units=16,
				 dropout_keep_ratio=0.5,
				 num_classes=14):
		super(PretrainedTextCNN, self).__init__()

		self.conv_layer = Conv1D(filters=filters, kernel_size=kernel_size)
		self.pool_layer = MaxPool1D()

		self.flatten_layer = Flatten()

		self.dense_layer = Dense(units=dense_units, activation='relu')

		dropout_keep_ratio = dropout_keep_ratio
		self.dropout_layer = Dropout(rate=dropout_keep_ratio)

		self.softmax_layer = Dense(units=num_classes, activation='softmax')


	def call(self, inputs, training=None, mask=None):
		# inputs: [None, seq_len, embedding_dim]
		# [None, conv_dim, embedding_dim], conv_dim = (W - F + 2P) / S + 1
		conv = self.conv_layer(inputs)

		# [None, pool_dim, embedding_dim], pool_dim = (W - F + 2P) / S + 1, S = pool_size = 2
		pool = self.pool_layer(conv)

		# [None, pool_dim * embedding_dim]
		flatten = self.flatten_layer(pool)

		# [None, dense_units]
		dense = self.dense_layer(flatten)
		dropout = self.dropout_layer(dense)

		# [None, num_classes]
		softmax = self.softmax_layer(dropout)
		return softmax


def test_model_once(model=None):
	batch_size = 4
	seq_len = 10
	embedding_dim = 300
	inputs = tf.random.uniform((batch_size, seq_len, embedding_dim))
	outputs = model(inputs)
	print('outputs.shape', outputs.shape)
	return outputs


if __name__ == '__main__':
	model = PretrainedTextCNN()
	test_model_once(model=model)

