import tensorflow as tf
tf.random.set_seed(7)
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, Concatenate


class PretrainedTextCNN(tf.keras.Model):
	def __init__(self,
				 filters=10,
				 kernel_size=5,
				 dense_units=16,
				 dropout_keep_ratio=0.5,
				 num_classes=14,
				 max_seq_len=350):
		super(PretrainedTextCNN, self).__init__()

		self.conv1_layer = Conv1D(filters=3, kernel_size=kernel_size, padding='same')
		self.conv2_layer = Conv1D(filters=4, kernel_size=kernel_size, padding='same')
		self.conv3_layer = Conv1D(filters=5, kernel_size=kernel_size, padding='same')

		self.pool_layer = MaxPool1D(pool_size=max_seq_len) # Time distributed pool

		self.concat_layer = Concatenate(axis=-1)

		self.flatten_layer = Flatten()

		self.dense_layer = Dense(units=dense_units, activation='relu')

		dropout_keep_ratio = dropout_keep_ratio
		self.dropout_layer = Dropout(rate=dropout_keep_ratio)

		self.softmax_layer = Dense(units=num_classes, activation='softmax')


	def call(self, inputs, training=None, mask=None):
		# inputs: [None, seq_len, embedding_dim]

		# [None, seq_len, filters], padding=same
		conv1 = self.conv1_layer(inputs)
		conv2 = self.conv2_layer(inputs)
		conv3 = self.conv3_layer(inputs)

		# [None, 1, filters]
		pool1 = self.pool_layer(conv1)
		pool2 = self.pool_layer(conv2)
		pool3 = self.pool_layer(conv3)

		# [None, 1, filters * 3]
		concat = self.concat_layer([pool1, pool2, pool3])

		# [None, filters * 3]
		flatten = self.flatten_layer(concat)

		# [None, dense_units]
		dense = self.dense_layer(flatten)
		dropout = self.dropout_layer(dense)

		# [None, num_classes]
		softmax = self.softmax_layer(dropout)
		return softmax


def test_model_once(model=None, max_seq_len=350):
	batch_size = 4
	#seq_len = 350
	embedding_dim = 300
	inputs = tf.random.uniform((batch_size, max_seq_len, embedding_dim))
	outputs = model(inputs)
	print('outputs.shape', outputs.shape)
	return outputs


if __name__ == '__main__':
	model = PretrainedTextCNN()
	test_model_once(model=model)

