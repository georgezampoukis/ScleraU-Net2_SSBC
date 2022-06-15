import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


def mean_iou(y_true, y_pred):
    y_pred = tf.keras.backend.cast(tf.keras.backend.greater(y_pred, 0.4), dtype='float32')  # .5 is the threshold
    inter = tf.keras.backend.sum(tf.keras.backend.sum(tf.keras.backend.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
    union = tf.keras.backend.sum(tf.keras.backend.sum(tf.keras.backend.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter

    return tf.keras.backend.mean((inter + tf.keras.backend.epsilon()) / (union + tf.keras.backend.epsilon()))


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

    return 1 - (numerator + 1) / (denominator + 1)


def Augmentor(x_data, y_data, batch_size, seed):

    data_gen_args = dict(rotation_range=20,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.1,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         fill_mode='nearest')

    x_generator = ImageDataGenerator(**data_gen_args)
    y_generator = ImageDataGenerator(**data_gen_args)

    x_generator.fit(x_data, augment=True, seed=seed)
    y_generator.fit(y_data, augment=True, seed=seed)

    x_augmented = x_generator.flow(x_data, batch_size=batch_size, seed=seed)
    y_augmented = y_generator.flow(y_data, batch_size=batch_size, seed=seed)

    generator = zip(x_augmented, y_augmented)

    return generator
