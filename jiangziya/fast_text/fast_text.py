import tensorflow as tf
from tensorflow.keras.layers import Embedding, Lambda
tf.random.set_seed(7)


class FastText(tf.keras.Model):
    def __init__(self,
                 vocab_size=10,
                 embedding_dim=4):
        super(FastText, self).__init__()

        self.embedding_layer = Embedding(input_dim=vocab_size+1,
                                         output_dim=embedding_dim,
                                         mask_zero=True)

    def call(self, inputs=None, input_lens=None):
        # inputs: [None, len]
        # embedding: [None, len, embedding_dim]
        embedding = self.embedding_layer(inputs)

        # [None, embedding_dim]
        embedding_sum = tf.reduce_sum(embedding, axis=1)

        # input_lens: [None, 1]
        # output: [None, embedding_dim]
        outputs = Lambda(divide_fn)([embedding_sum, input_lens])

        return outputs


def divide_fn(inputs):
    x, y = inputs
    return tf.divide(x, y)


if __name__ == '__main__':
    ft = FastText()
    inputs = tf.constant([[1, 2, 3, 0], [1, 2, 0, 0]], dtype=tf.int32)
    input_lens = tf.constant([[3], [2]], dtype=tf.float32)
    outputs = ft(inputs=inputs, input_lens=input_lens)

    print(outputs)