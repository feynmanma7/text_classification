from jiangziya.fast_text.fast_text import PretrainedFastText
from jiangziya.utils.config import get_model_dir, get_data_dir, get_log_dir
from jiangziya.fast_text.dataset import get_pretrained_dataset
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import time
import os


def train_pretrained_fast_text():
    total_num_train = 69000 # num_lines of shuf_train_seg
    total_num_val = 17300 # num_lines of shuf_val_seg

    epochs = 100
    shuffle_buffer_size = 1024 * 2
    batch_size = 32
    num_classes = 14
    patience = 10 # for early stopping

    data_dir = os.path.join(get_data_dir(), "text_classification")
    train_path = os.path.join(data_dir, "thucnews_train_vec.txt")
    val_path = os.path.join(data_dir, "thucnews_val_vec.txt")

    log_dir = os.path.join(get_log_dir(), "fast_text")
    checkpoint_path = os.path.join(get_model_dir(), "fast_text", "ckpt")

    num_train_batch = total_num_train // batch_size + 1
    num_val_batch = total_num_val // batch_size + 1

    # === tf.data.Dataset
    train_dataset = get_pretrained_dataset(data_path=train_path,
                                epochs=epochs,
                                shuffle_buffer_size=shuffle_buffer_size,
                                batch_size=batch_size)

    val_dataset = get_pretrained_dataset(data_path=val_path,
                              epochs=epochs,
                              shuffle_buffer_size=shuffle_buffer_size,
                              batch_size=batch_size)

    # === model
    model = PretrainedFastText(num_classes=num_classes)

    # optimizer
    optimizer = tf.keras.optimizers.Adam(0.001)

    # loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(optimizer=optimizer,
                   loss=loss,
                   metrics=['acc'])

    # callbacks
    callbacks = []

    early_stopping_cb = EarlyStopping(monitor='val_loss',
                                      patience=patience,
                                      restore_best_weights=True)
    callbacks.append(early_stopping_cb)

    tensorboard_cb = TensorBoard(log_dir=log_dir)
    callbacks.append(tensorboard_cb)

    checkpoint_cb = ModelCheckpoint(filepath=checkpoint_path,
                                    save_weights_only=True,
                                    save_best_only=True)
    callbacks.append(checkpoint_cb)

    # === Train
    print(model.summary())

    history = model.fit(train_dataset,
               epochs=epochs,
               steps_per_epoch=num_train_batch,
               validation_data=val_dataset,
               validation_steps=num_val_batch,
               callbacks=callbacks)

    return history


if __name__ == "__main__":
    start = time.time()
    train_pretrained_fast_text()
    end = time.time()
    last = end - start
    print("\nTrain done! Lasts: %.2fs" % last)