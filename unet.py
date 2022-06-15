import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, SpatialDropout2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
import tensorflow_addons as tfa
from tensorflow_addons.layers import GroupNormalization
from tensorflow_addons.activations import gelu
from tensorflow.keras.optimizers import Adam
import TrainFuncs


def unet(input_img, n_filters, dropout):

    inputs = Input(input_img)

    #Downspample
    c1 = Conv2D(filters=n_filters, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(inputs)
    c1 = gelu(c1)
    c1 = GroupNormalization(groups=16, axis=3)(c1)
    c1 = Conv2D(filters=n_filters, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(c1)
    c1 = gelu(c1)
    c1 = GroupNormalization(groups=16, axis=3)(c1)
    d1 = SpatialDropout2D(dropout)(c1)
    p1 = MaxPooling2D((2, 2))(d1)
    
    c2 = Conv2D(filters=n_filters * 2, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(p1)
    c2 = gelu(c2)
    c2 = GroupNormalization(groups=16, axis=3)(c2)
    c2 = Conv2D(filters=n_filters * 2, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(c2)
    c2 = gelu(c2)
    c2 = GroupNormalization(groups=16, axis=3)(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(filters=n_filters * 4, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(p2)
    c3 = gelu(c3)
    c3 = GroupNormalization(groups=16, axis=3)(c3)
    c3 = Conv2D(filters=n_filters * 4, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(c3)
    c3 = gelu(c3)
    c3 = GroupNormalization(groups=16, axis=3)(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(filters=n_filters * 8, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(p3)
    c4 = gelu(c4)
    c4 = GroupNormalization(groups=16, axis=3)(c4)
    c4 = Conv2D(filters=n_filters * 8, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(c4)
    c4 = gelu(c4)
    c4 = GroupNormalization(groups=16, axis=3)(c4)
    d4 = SpatialDropout2D(dropout)(c4)
    p4 = MaxPooling2D((2, 2))(d4)

    c5 = Conv2D(filters=n_filters * 16, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(p4)
    c5 = gelu(c5)
    c5 = GroupNormalization(groups=16, axis=3)(c5)
    c5 = Conv2D(filters=n_filters * 16, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(c5)
    d5 = gelu(c5)
    c5 = GroupNormalization(groups=16, axis=3)(c5)
    d5 = SpatialDropout2D(dropout)(d5)

    # Upsample
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(d5)
    u6 = concatenate([u6, c4])
    u6 = GroupNormalization(groups=16, axis=3)(u6)
    c6 = Conv2D(filters=n_filters * 8, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(u6)
    c6 = gelu(c6)
    c6 = GroupNormalization(groups=16, axis=3)(c6)
    c6 = Conv2D(filters=n_filters * 8, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(c6)
    c6 = gelu(c6)
    c6 = GroupNormalization(groups=16, axis=3)(c6)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = GroupNormalization(groups=16, axis=3)(u7)
    c7 = Conv2D(filters=n_filters * 4, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(u7)
    c7 = gelu(c7)
    c7 = GroupNormalization(groups=16, axis=3)(c7)
    c7 = Conv2D(filters=n_filters * 4, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(c7)
    c7 = gelu(c7)
    c7 = GroupNormalization(groups=16, axis=3)(c7)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = GroupNormalization(groups=16, axis=3)(u8)
    c8 = Conv2D(filters=n_filters * 2, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(u8)
    c8 = gelu(c8)
    c8 = GroupNormalization(groups=16, axis=3)(c8)
    c8 = Conv2D(filters=n_filters * 2, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(c8)
    c8 = gelu(c8)
    c8 = GroupNormalization(groups=16, axis=3)(c8)

    u9 = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = GroupNormalization(groups=16, axis=3)(u9)
    c9 = Conv2D(filters=n_filters, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(u9)
    c9 = gelu(c9)
    c9 = GroupNormalization(groups=16, axis=3)(c9)
    c9 = Conv2D(filters=n_filters, kernel_size=(3, 3), kernel_initializer="he_normal", padding="same")(c9)
    c9 = gelu(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=Adam(learning_rate=1e-3), loss=[TrainFuncs.dice_loss], metrics=[TrainFuncs.mean_iou])

    # model.summary()

    return model
