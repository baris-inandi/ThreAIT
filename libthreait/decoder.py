from keras import layers
from tensorflow import keras


def make_model_decoder(encoded_shape, output_shape):
    inputs = keras.Input(shape=(encoded_shape,))
    x = inputs

    encoded_dim = output_shape[0] // 8
    x = layers.Dense(encoded_dim * encoded_dim * 128, activation="relu")(x)
    x = layers.Reshape((encoded_dim, encoded_dim, 128))(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)

    x = layers.Conv2DTranspose(
        output_shape[2], 3, padding="same", activation="sigmoid"
    )(x)
    x = layers.Rescaling(255)(x)
    outputs = x
    return keras.Model(inputs, outputs)
