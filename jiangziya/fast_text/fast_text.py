import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
tf.random.set_seed(7)


class PretrainedFastText(tf.keras.Model):
    def __init__(self, num_classes=14):
        super(PretrainedFastText, self).__init__()

        dropout_rate = 0.2

        self.dense_1_layer = Dense(units=128, activation='relu')
        self.dropout_1 = Dropout(dropout_rate)

        self.softmax_layer = Dense(units=num_classes, activation='softmax')

    def call(self, inputs=None):
        # inputs: [None, 300]

        # [None, 128]
        outputs = self.dense_1_layer(inputs)
        outputs = self.dropout_1(outputs)

        # [None, num_classes]
        outputs = self.softmax_layer(outputs)

        return outputs


def test_pretrained_fast_text_once(model=None):
    batch_size = 2
    embedding_dim = 300
    # [2, 300]
    inputs = tf.random.uniform((batch_size, embedding_dim))

    # [2, num_classes]
    outputs = model(inputs)
    return outputs


if __name__ == '__main__':
    ft = PretrainedFastText(num_classes=14)
    outputs = test_pretrained_fast_text_once(model=ft)
    print(outputs.shape)