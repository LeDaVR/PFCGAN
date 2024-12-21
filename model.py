import tensorflow as tf
from tensorflow.keras import layers
from utils import mask_rgb
import numpy as np

## Create the models

def make_extractor_model():
    input_image = layers.Input(shape=(128, 128, 5))
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='leaky_relu')(input_image)    
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='leaky_relu')(x)    
    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='leaky_relu')(x)    
    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='leaky_relu')(x)    
    x = layers.Flatten()(x)

    out_dim = 512

    z_mean = layers.Dense(out_dim, kernel_initializer='zeros', name="z_mean")(x)
    z_log_var = tf.clip_by_value(layers.Dense(out_dim, kernel_initializer='zeros', name="z_log_var")(x), -10, 10)
    
    # Model outputs
    return tf.keras.Model(inputs=[input_image], outputs=[z_mean, z_log_var], name="extractor")

### The Generator

def make_generator_model():
    l_dim = 512
    input_latent = layers.Input(shape=(l_dim,))
    input_image = layers.Input(shape=(128, 128, 3))
    mask = layers.Input(shape=(128, 128, 1))

    # Procesar el vector latente
    y = layers.Dense(8*8*512)(input_latent)
    y = layers.Reshape((8, 8, 512))(y)
    # Concatenar la imagen de entrada si es necesario    
    # ones_x = layers.Lambda(lambda x: tf.ones_like(x)[:, :, :, 0:1])(input_image)  # Fix the operation with Lambda layer
    # print("mask", tf.shape(mask), tf.shape(ones_x))
    x = layers.Concatenate(axis=-1, name="concat_mask")([input_image, mask])
    x = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)    
    x = c1 = layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)    
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = c2 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same' , activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = c3 = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = c4 = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)



    x = layers.Concatenate(axis=-1)([x,y])
    x = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Concatenate(axis=-1)([x,c4])
    x = layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Concatenate(axis=-1)([x,c3])
    x = layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Concatenate(axis=-1)([x,c2])
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
    x = layers.Concatenate(axis=-1)([x,c1])
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    output = layers.Conv2D(3, (3, 3), strides=(1, 1), padding='same', activation='tanh')(x)
    # Crear modelo con dos inputs
    model = tf.keras.Model(inputs=[input_latent, input_image, mask], outputs=output, name="generator")
    
    return model

def make_discriminator_model():
    input_image = layers.Input(shape=(128, 128, 3))
    x = layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', activation='leaky_relu')(input_image)
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='leaky_relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    return tf.keras.Model(inputs=[input_image], outputs=x, name='global_discriminator')

def make_landmark_encoder():
    z_dim = 512
    filters = 32

    #Inputs
    mask = layers.Input(shape=(128,128,1))
    incomplete = layers.Input(shape=(128,128,3))
    # z = layers.Input(shape=(z_dim,))

    # z_dense = layers.Dense(4096)(z)
    # z1 = layers.Reshape((64, 64, 1))(z_dense)
    # z2 = layers.Reshape((32, 32, -1))(z1)
    # z3 = layers.Reshape((16, 16, -1))(z2)
    # z4 = layers.Reshape((8, 8, -1))(z3)
    # z5 = layers.Reshape((4, 4, -1))(z4)

    x = layers.Concatenate(axis=-1)([incomplete, mask])
    # concatenate reshaped z between convolutions
    x = layers.Conv2D(filters, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    # x = layers.Concatenate(axis=-1)([x, z1])
    x = layers.Conv2D(filters * 2, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    # x = layers.Concatenate(axis=-1)([x, z2])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    # x = layers.Concatenate(axis=-1)([x, z3])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    # x = layers.Concatenate(axis=-1)([x, z4])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    # x = layers.Concatenate(axis=-1)([x, z5])
    x = layers.Flatten()(x)
    # x = layers.Concatenate(axis=-1)([x, z])

    out_dim = 256
    z_mean = layers.Dense(out_dim, kernel_initializer='zeros')(x)
    z_log_var = tf.clip_by_value(layers.Dense(out_dim, kernel_initializer='zeros')(x), -10, 10)

    model = tf.keras.Model(inputs=[incomplete, mask], outputs=[z_mean, z_log_var], name="landmark_encoder")

    return model

# Same as landmark encoder
def make_face_encoder():
    z_dim = 512
    filters = 32

    #Inputs
    incomplete = layers.Input(shape=(128,128,3))
    mask = layers.Input(shape=(128,128,1))
    # z = layers.Input(shape=(z_dim,))


    # z_dense = layers.Dense(4096)(z)
    # z1 = layers.Reshape((64, 64, 1))(z_dense)
    # z2 = layers.Reshape((32, 32, -1))(z1)
    # z3 = layers.Reshape((16, 16, -1))(z2)
    # z4 = layers.Reshape((8, 8, -1))(z3)
    # z5 = layers.Reshape((4, 4, -1))(z4)

    # concatenate incomplete and mask
    x = layers.Concatenate(axis=-1)([incomplete, mask])
    # concatenate reshaped z between convolutions
    x = layers.Conv2D(filters, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    # x = layers.Concatenate(axis=-1)([x, z1])
    x = layers.Conv2D(filters * 2, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    # x = layers.Concatenate(axis=-1)([x, z2])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    # x = layers.Concatenate(axis=-1)([x, z3])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    # x = layers.Concatenate(axis=-1)([x, z4])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    # x = layers.Concatenate(axis=-1)([x, z5])
    x = layers.Flatten()(x)
    # x = layers.Concatenate(axis=-1)([x, z])

    out_dim = 256
    z_mean = layers.Dense(out_dim, kernel_initializer='zeros')(x)
    z_log_var = tf.clip_by_value(layers.Dense(out_dim, kernel_initializer='zeros')(x), -10, 10)

    model = tf.keras.Model(inputs=[incomplete, mask], outputs=[z_mean, z_log_var], name="landmark_encoder")

    return model

def make_landmark_decoder():
    input_dim = 256 + 512
    z = layers.Input(shape=(input_dim,))
    filters = 32

    reshape_channels = input_dim // 64
    x = layers.Reshape((8, 8, reshape_channels))(z)
    x = layers.Conv2DTranspose(filters * 8, (4,4), (2,2), name = 'deconv1', activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(filters * 4, (4,4), (2,2), name = 'deconv2', activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(filters * 2, (4,4), (2,2), name = 'deconv3', activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(filters, (4,4), (2,2), name = 'deconv4', activation='relu', padding='same')(x)
    x = layers.Conv2D(1, (3,3), (1,1), name='conv1', padding='same')(x)
    model = tf.keras.Model(inputs=[z], outputs=x, name="landmark_decoder")

    return model

# Same as landmark decoder but input 512 and output 3 channels
def make_face_part_decoder():
    input_dim = 512 + 512
    z = layers.Input(shape=(input_dim,))
    filters = 32

    reshape_channels = input_dim // 64
    x = layers.Reshape((8, 8, reshape_channels))(z)
    x = layers.Conv2DTranspose(filters * 8, (4,4), (2,2), name = 'deconv1', activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(filters * 4, (4,4), (2,2), name = 'deconv2', activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(filters * 2, (4,4), (2,2), name = 'deconv3', activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(filters, (4,4), (2,2), name = 'deconv4', activation='relu', padding='same')(x)
    x = layers.Conv2D(3, (3,3), (1,1), activation=tf.nn.tanh, name='conv1', padding='same')(x)

    model = tf.keras.Model(inputs=[z], outputs=x, name="face_part_decoder")

    return model

# Same network as face mask decoder
def make_face_mask_decoder():
    input_dim = 512 + 512
    z = layers.Input(shape=(input_dim,))
    filters = 32

    reshape_channels = input_dim // 64
    x = layers.Reshape((8, 8, reshape_channels))(z)
    x = layers.Conv2DTranspose(filters * 8, (4,4), (2,2), name = 'deconv1', activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(filters * 4, (4,4), (2,2), name = 'deconv2', activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(filters * 2, (4,4), (2,2), name = 'deconv3', activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(filters, (4,4), (2,2), name = 'deconv4', activation='relu', padding='same')(x)
    x = layers.Conv2D(1, (3,3), (1,1), name='conv1', padding='same')(x)

    model = tf.keras.Model(inputs=[z], outputs=x, name="face_mask_decoder")

    return model

def make_local_discriminator():
    input_image = layers.Input(shape=(128, 128, 3))
    mask = layers.Input(shape=(128, 128, 1))
    features =  32
    x = layers.concatenate([input_image, mask], axis=-1)
    x = layers.Conv2D(features, (5, 5), strides=(2, 2), padding='same',activation='leaky_relu')(x)
    x = layers.Conv2D(features * 2, (5, 5), strides=(2, 2), padding='same',activation='leaky_relu')(x)
    x = layers.Conv2D(features * 4, (5, 5), strides=(2, 2), padding='same',activation='leaky_relu')(x)
    x = layers.Conv2D(features * 8, (5, 5), strides=(2, 2), padding='same',activation='leaky_relu')(x)
    x = layers.Conv2D(features * 8, (5, 5), strides=(2, 2), padding='same',activation='leaky_relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=[input_image, mask], outputs=x, name="local_discriminator")

    return model

class CyclicalAnnealingScheduler:
    def __init__(self, cycle_length, max_beta=1.0, min_beta=0.0, n_cycles=4):
        self.cycle_length = cycle_length
        self.max_beta = max_beta
        self.min_beta = min_beta
        self.n_cycles = n_cycles
        
    def get_beta(self, epoch):
        """Calcula el valor beta para el epoch actual"""
        # Determinar el ciclo actual
        cycle = (epoch % self.cycle_length) / self.cycle_length
        
        # Calcular el valor beta usando una función suave
        beta = self.min_beta + (self.max_beta - self.min_beta) * \
               (1 / (1 + np.exp(-12 * (cycle - 0.5))))
        
        return tf.cast(beta, dtype=tf.float32)
