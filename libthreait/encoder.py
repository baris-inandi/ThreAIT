import tensorflow as tf
from keras import layers
from tensorflow import keras


class VarSampling(keras.layers.Layer):
    def __call__(self, inputs):
        encoded_mean, encoded_log_var = inputs
        batch = tf.shape(encoded_mean)[0]
        dim = tf.shape(encoded_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return encoded_mean + tf.exp(0.5 * encoded_log_var) * epsilon


def make_model_encoder(input_shape, encoded_shape):
    use_max_pooling = False
    dropout_shape = 0.3
    activation = "relu"  # or use another one from "https://www.tensorflow.org/api_docs/python/tf/keras/activations"

    # Instead of using strides, you can use max-pooling layers for downsampling,
    inputs = layers.Input(shape=input_shape)
    x = inputs
    base_model = tf.keras.applications.EfficientNetV2M(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling=None,
        classes=encoded_shape*2,
        classifier_activation="softmax",
        include_preprocessing=True,
    )
    base_model.trainable = False
    x = base_model(x)
    # this is the max pooling downsampling
    x = layers.Conv2D(256,3,padding="same",activation=activation)(x)
    x = layers.Conv2D(128,3,padding="same",activation=activation)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout_shape)(x)

    encoded_mean = layers.Dense(encoded_shape)(x)
    encoded_log_var = layers.Dense(encoded_shape)(x)
    encoded = VarSampling()([encoded_mean, encoded_log_var])

    return keras.Model(inputs, [encoded_mean, encoded_log_var, encoded])
