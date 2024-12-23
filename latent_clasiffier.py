import tensorflow as tf
from tensorflow.keras import layers, Model

# Definimos el clasificador latente
class LatentClassifier(Model):
    def __init__(self, latent_dim):
        super(LatentClassifier, self).__init__()
        self.dense1 = layers.Dense(256, activation='relu')  # Más neuronas
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(64, activation='relu')   # Capa adicional
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.output_layer(x)


# Función de pérdida para el clasificador latente
def latent_discriminator_loss(real_pred, fake_pred):
    # -E[log Cw(V)] - E[log(1 - Cw(V_generated))]
    eps = tf.keras.backend.epsilon()
    loss = -(tf.reduce_mean(tf.math.log(real_pred + eps)) + 
            tf.reduce_mean(tf.math.log(1 - fake_pred + eps)))
    return loss
    # Clasificación correcta de z_real como verdadero
    epsilon = 1e-8  # Valor pequeño para evitar log(0) o log(1)

    # Clasificación real de z_real
    log_C_real = tf.reduce_mean(tf.math.log(tf.clip_by_value(classifier(z_real), epsilon, 1.0)))

    # Clasificación correcta de z_fake como falso
    log_1_minus_C_fake = tf.reduce_mean(tf.math.log(tf.clip_by_value(1 - classifier(z_fake), epsilon, 1.0)))

    # Pérdida total
    latent_loss = -(log_C_real + log_1_minus_C_fake)
    return latent_loss

def latent_generator_loss(fake_output):
    eps = tf.keras.backend.epsilon()
    return -tf.reduce_mean(tf.math.log(fake_output + eps))
