import tensorflow as tf
from tensorflow.keras import layers
from utils import mask_rgb

## Create the models

def make_extractor_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[128, 128, 8]))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())

    out_dim = 512

    z_mean = layers.Dense(out_dim, name="z_mean")(model.output)
    z_log_var = layers.Dense(out_dim, name="z_log_var")(model.output)
    
    # Model outputs
    return tf.keras.Model(inputs=model.input, outputs=[z_mean, z_log_var], name="extractor")

### The Generator

def make_generator_model():
    l_dim = 512
    input_latent = layers.Input(shape=(l_dim,))
    input_image = layers.Input(shape=(128, 128, 3))
    mask = layers.Input(shape=(128, 128, 3))

    # Procesar el vector latente
    y = layers.Dense(8*8*512)(input_latent)
    y = layers.Reshape((8, 8, 512))(y)
    # Concatenar la imagen de entrada si es necesario    
    ones_x = tf.ones_like(input_image)[:, :, :, 0:1]
    # print("mask", tf.shape(mask), tf.shape(ones_x))
    x = layers.Concatenate(axis=-1, name="concat_mask")([input_image, ones_x *mask ])
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
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[128, 128, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def make_landmark_encoder():
    z_dim = 512
    filters = 32
    incomplete = layers.Input(shape=(128,128,3))
    z = layers.Input(shape=(z_dim,))
    z_dense = layers.Dense(4096)(z)
    z1 = layers.Reshape((64, 64, 1))(z_dense)
    z2 = layers.Reshape((32, 32, -1))(z1)
    z3 = layers.Reshape((16, 16, -1))(z2)
    z4 = layers.Reshape((8, 8, -1))(z3)
    z5 = layers.Reshape((4, 4, -1))(z4)

    # concatenate reshaped z between convolutions
    x = layers.Conv2D(filters, (4,4), (2,2), padding='same',activation='elu')(incomplete)
    x = layers.Concatenate(axis=-1)([x, z1])
    x = layers.Conv2D(filters * 2, (4,4), (2,2), padding='same',activation='elu')(x)
    x = layers.Concatenate(axis=-1)([x, z2])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='elu')(x)
    x = layers.Concatenate(axis=-1)([x, z3])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='elu')(x)
    x = layers.Concatenate(axis=-1)([x, z4])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='elu')(x)
    x = layers.Concatenate(axis=-1)([x, z5])
    x = layers.Flatten()(x)

    out_dim = 256
    z_mean = layers.Dense(out_dim)(x)
    z_log_var = layers.Dense(out_dim)(x)

    model = tf.keras.Model(inputs=[incomplete, z], outputs=[z_mean, z_log_var], name="landmark_encoder")

    return model

# Same as landmark encoder
def make_face_encoder():
    return make_landmark_encoder()

def make_landmark_decoder():
    input_dim = 256
    z = layers.Input(shape=(256,))
    filters = 32

    reshape_channels = input_dim // 64
    x = layers.Reshape((8, 8, reshape_channels))(z)
    x = layers.Conv2DTranspose(filters * 8, (4,4), (2,2), name = 'deconv1', padding='same')(x)
    x = layers.Conv2DTranspose(filters * 4, (4,4), (2,2), name = 'deconv2', padding='same')(x)
    x = layers.Conv2DTranspose(filters * 2, (4,4), (2,2), name = 'deconv3', padding='same')(x)
    x = layers.Conv2DTranspose(filters, (4,4), (2,2), name = 'deconv4', padding='same')(x)
    x = layers.Conv2D(1, (3,3), (1,1), activation=tf.nn.tanh, name='conv1', padding='same')(x)

    model = tf.keras.Model(inputs=[z], outputs=x, name="landmark_decoder")

    return model

# Same as landmark decoder but input 512 and output 3 channels
def make_face_part_decoder():
    input_dim = 512
    z = layers.Input(shape=(input_dim,))
    filters = 32

    reshape_channels = input_dim // 64
    x = layers.Reshape((8, 8, reshape_channels))(z)
    x = layers.Conv2DTranspose(filters * 8, (4,4), (2,2), name = 'deconv1', padding='same')(x)
    x = layers.Conv2DTranspose(filters * 4, (4,4), (2,2), name = 'deconv2', padding='same')(x)
    x = layers.Conv2DTranspose(filters * 2, (4,4), (2,2), name = 'deconv3', padding='same')(x)
    x = layers.Conv2DTranspose(filters, (4,4), (2,2), name = 'deconv4', padding='same')(x)
    x = layers.Conv2D(3, (3,3), (1,1), activation=tf.nn.tanh, name='conv1', padding='same')(x)

    model = tf.keras.Model(inputs=[z], outputs=x, name="face_part_decoder")

    return model

# Same network as face mask decoder
def make_face_mask_decoder():
    input_dim = 512
    z = layers.Input(shape=(input_dim,))
    filters = 32

    reshape_channels = input_dim // 64
    x = layers.Reshape((8, 8, reshape_channels))(z)
    x = layers.Conv2DTranspose(filters * 8, (4,4), (2,2), name = 'deconv1', padding='same')(x)
    x = layers.Conv2DTranspose(filters * 4, (4,4), (2,2), name = 'deconv2', padding='same')(x)
    x = layers.Conv2DTranspose(filters * 2, (4,4), (2,2), name = 'deconv3', padding='same')(x)
    x = layers.Conv2DTranspose(filters, (4,4), (2,2), name = 'deconv4', padding='same')(x)
    x = layers.Conv2D(1, (3,3), (1,1), activation=tf.nn.tanh, name='conv1', padding='same')(x)

    model = tf.keras.Model(inputs=[z], outputs=x, name="face_part_decoder")

    return model
