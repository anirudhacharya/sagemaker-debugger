# Standard Library
import datetime as dt

# Third Party
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

train_dataset, valid_dataset = tfds.load(
    "cifar10", split=["train", "test"], batch_size=8, as_supervised=True
)
# (x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load('imagenette', split=['train', 'validation'], batch_size=64, as_supervised=True,))

# train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
train_dataset = train_dataset.map(lambda x, y: (tf.image.central_crop(x, 0.75), y))
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
train_dataset = train_dataset.repeat()

# valid_dataset = valid_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
valid_dataset = valid_dataset.map(lambda x, y: (tf.image.central_crop(x, 0.75), y))
valid_dataset = valid_dataset.repeat()


def res_net_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation="relu", padding="same")(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=None, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_data])
    x = layers.Activation("relu")(x)
    return x


def non_res_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation="relu", padding="same")(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    return x


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    inputs = keras.Input(shape=(24, 24, 3))
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)

    num_res_net_blocks = 10
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)

    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    res_net_model = keras.Model(inputs, outputs)
    opt = tf.optimizers.Adam()
    res_net_model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
        experimental_run_tf_function=False,
    )


def lr_decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


# Create a TensorBoard callback
logs = "logs_mirrored_strat/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs, histogram_freq=1, profile_batch=2)

# create leraning rate scheduler callback.
lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay)

callbacks = [tboard_callback, lr_scheduler_callback]

res_net_model.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=3500,
    validation_data=valid_dataset,
    validation_steps=250,
    callbacks=callbacks,
)
