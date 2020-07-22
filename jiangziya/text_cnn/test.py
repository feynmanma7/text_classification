import tensorflow as tf
tf.random.set_seed(7)
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense

if __name__ == '__main__':
    batch_size = 2
    seq_len = 10
    embedding_dim = 8
    inputs = tf.random.uniform((batch_size, seq_len, embedding_dim))
    print(inputs.shape)

    conv = Conv1D(filters=2, kernel_size=3, padding='same')(inputs)
    print(conv.shape)

    pool = MaxPool1D(pool_size=7)(conv)
    print(pool.shape)

    pool = MaxPool1D(pool_size=6)(conv)
    print(pool.shape)
