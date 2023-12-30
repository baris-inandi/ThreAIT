import tensorflow as tf
from tensorflow import keras


class ThreAIT(keras.Model):
    def __init__(self, encoder, decoder):
        super(ThreAIT, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.counter = 0

    def compile(self, ae_optimizer, loss_fn):
        super(ThreAIT, self).compile()
        self.ae_optimizer = ae_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_events):
        if isinstance(real_events, tuple):
            real_events = real_events[0]
        with tf.GradientTape() as tape:
            encoded_mean, encoded_log_var, encoded = self.encoder(real_events)
            reconstruction = self.decoder(encoded)
            #reconstruction_loss = 0
            if(self.counter < 10):
                reconstruction_loss = tf.reduce_mean(
                    self.loss_fn(real_events, reconstruction)
                )
            else:
                reconstruction_loss = tf.reduce_mean(
                    self.loss_fn(real_events, tf.round(reconstruction))
                )
            self.counter+= 1
            if(self.counter == 10):
                print("I am at the 10th step")
            # reconstruction_loss *= tf.reduce_prod(tf.shape(real_events)[1:])
            kl_loss = (
                1 + encoded_log_var - tf.square(encoded_mean) - tf.exp(encoded_log_var)
            )
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, real_events):
        if isinstance(real_events, tuple):
            real_events = real_events[0]

        encoded_mean, encoded_log_var, encoded = self.encoder(real_events)
        reconstruction = self.decoder(encoded)
        reconstruction_loss = tf.reduce_mean(self.loss_fn(real_events, reconstruction))
        # reconstruction_loss *= tf.reduce_prod(tf.shape(real_events)[1:])
        kl_loss = (
            1 + encoded_log_var - tf.square(encoded_mean) - tf.exp(encoded_log_var)
        )
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }
