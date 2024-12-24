import tensorflow as tf
from tensorflow.keras import layers, Model

# Definimos el clasificador latente
class LatentClassifier(Model):
    def __init__(self, latent_dim):
        super(LatentClassifier, self).__init__()
        self.dense1 = layers.Dense(256, activation='leaky_relu')
        self.dense2 = layers.Dense(128, activation='leaky_relu')
        self.dense3 = layers.Dense(64, activation='leaky_relu')   # Capa adicional
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)


# Función de pérdida para el clasificador latente
def latent_discriminator_loss(real_pred, fake_pred):
    eps = tf.keras.backend.epsilon()
    real_loss = -tf.reduce_mean(tf.math.log(real_pred + eps ))
    fake_loss = -tf.reduce_mean(tf.math.log(1. - fake_pred + eps ))
    return (real_loss + fake_loss)

def latent_generator_loss(fake_output):
    eps = tf.keras.backend.epsilon()
    return -tf.reduce_mean(tf.math.log(fake_output + eps))
