import tensorflow as tf
tf.random.set_seed(7)
from tensorflow.keras.layers import Embedding, Conv1D, MaxPool1D, Dense, Dropout, Flatten


class TextCNN(tf.keras.Model):
	def __init__(self, vocab_size=10,
				 num_classes=14,
				 embedding_dim=16,
				 filters=5,
				 kernel_size=3,
				 dense_units=8):
		super(TextCNN, self).__init__()

		dropout_keep_ratio = 0.5
		self.embedding_dim = embedding_dim

		# index = 0, for `pad`, index = 1, for `unk`
		self.embedding_layer = Embedding(input_dim=vocab_size+2,
										 output_dim=embedding_dim,
										 mask_zero=True)
		self.conv_layer = Conv1D(filters=filters, kernel_size=kernel_size)
		self.max_pool_layer = MaxPool1D()

		self.flatten_layer = Flatten()
		self.dense_layer = Dense(units=dense_units, activation='relu')
		self.dropout_layer = Dropout(rate=dropout_keep_ratio)

		self.softmax_layer = Dense(units=num_classes, activation='softmax')


	def call(self, inputs, training=None, mask=None):
		# === Embedding
		# inputs: [None, seq_len]
		# embedding: [None, seq_len, embedding_dim]
		embedding = self.embedding_layer(inputs)

		# === Mask Zero of Embedding.
		# [None, seq_len]
		masked_embedding = tf.cast(embedding._keras_mask, tf.float32)

		# [None, seq_len * embedding_dim]
		masked_embedding = tf.repeat(masked_embedding, repeats=self.embedding_dim, axis=1)

		# [None, seq_len, embedding_dim]
		masked_embedding = tf.reshape(masked_embedding,
									  shape=[-1, inputs.shape[1], self.embedding_dim])

		# [None, seq_len, embedding_dim]
		embedding = tf.multiply(embedding, masked_embedding)

		# === Conv
		# [None, conv_dim, embedding_dim], conv_dim = (W - F + 2P) / S + 1
		conv = self.conv_layer(embedding)

		# === Pool
		# [None, pad_dim, embedding_dim], pad_dim = (W - F + 2P) / S + 1
		pool = self.max_pool_layer(conv)

		# === Dense
		# [None, pad_dim * embedding_dim]
		flatten = self.flatten_layer(pool)

		# [None, dense_units]
		dense = self.dense_layer(flatten)
		dropout = self.dropout_layer(dense)

		# === Softmax
		# [None, num_classes]
		softmax = self.softmax_layer(dropout)

		return softmax


def test_model_once(model=None, vocab_size=None):
	batch_size = 4
	seq_len = 10
	inputs = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 0, 0],
						  [2, 4, 6, 8, 10, 1, 3, 0, 0, 0]], dtype=tf.int32)
	outputs = model(inputs)
	print('outputs.shape', outputs.shape)


if __name__ == '__main__':
	vocab_size = 10
	model = TextCNN(vocab_size=vocab_size)

	test_model_once(model=model, vocab_size=vocab_size)

	model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
				  loss=tf.keras.losses.SparseCategoricalCrossentropy,
				  metrics=['acc'])
	print(model.summary())
