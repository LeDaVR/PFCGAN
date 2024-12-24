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
    z = layers.Input(shape=(z_dim,))

    z_dense = layers.Dense(4096)(z)
    z1 = layers.Reshape((64, 64, 1))(z_dense)
    z2 = layers.Reshape((32, 32, -1))(z1)
    z3 = layers.Reshape((16, 16, -1))(z2)
    # z4 = layers.Reshape((8, 8, -1))(z3)
    # z5 = layers.Reshape((4, 4, -1))(z4)

    x = layers.Concatenate(axis=-1)([incomplete, mask])
    # concatenate reshaped z between convolutions
    x = layers.Conv2D(filters, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    x = layers.Concatenate(axis=-1)([x, z1])
    x = layers.Conv2D(filters * 2, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    x = layers.Concatenate(axis=-1)([x, z2])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    x = layers.Concatenate(axis=-1)([x, z3])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    # x = layers.Concatenate(axis=-1)([x, z4])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    # x = layers.Concatenate(axis=-1)([x, z5])
    x = layers.Flatten()(x)
    x = layers.Concatenate(axis=-1)([x, z])

    out_dim = 256
    z_mean = layers.Dense(out_dim, kernel_initializer='zeros')(x)
    z_log_var = tf.clip_by_value(layers.Dense(out_dim, kernel_initializer='zeros')(x), -10, 10)

    model = tf.keras.Model(inputs=[incomplete, z, mask], outputs=[z_mean, z_log_var], name="landmark_encoder")

    return model

# Same as landmark encoder
def make_face_encoder():
    z_dim = 512
    filters = 32

    #Inputs
    incomplete = layers.Input(shape=(128,128,3))
    mask = layers.Input(shape=(128,128,1))
    z = layers.Input(shape=(z_dim,))


    z_dense = layers.Dense(4096)(z)
    z1 = layers.Reshape((64, 64, 1))(z_dense)
    z2 = layers.Reshape((32, 32, -1))(z1)
    z3 = layers.Reshape((16, 16, -1))(z2)
    # z4 = layers.Reshape((8, 8, -1))(z3)
    # z5 = layers.Reshape((4, 4, -1))(z4)

    # concatenate incomplete and mask
    x = layers.Concatenate(axis=-1)([incomplete, mask])
    # concatenate reshaped z between convolutions
    x = layers.Conv2D(filters, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    x = layers.Concatenate(axis=-1)([x, z1])
    x = layers.Conv2D(filters * 2, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    x = layers.Concatenate(axis=-1)([x, z2])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    x = layers.Concatenate(axis=-1)([x, z3])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    # x = layers.Concatenate(axis=-1)([x, z4])
    x = layers.Conv2D(filters * 4, (4,4), (2,2), padding='same',activation='leaky_relu')(x)
    # x = layers.Concatenate(axis=-1)([x, z5])
    x = layers.Flatten()(x)
    # x = layers.Concatenate(axis=-1)([x, z])

    out_dim = 256
    z_mean = layers.Dense(out_dim, kernel_initializer='zeros')(x)
    z_log_var = tf.clip_by_value(layers.Dense(out_dim, kernel_initializer='zeros')(x), -10, 10)

    model = tf.keras.Model(inputs=[incomplete, z, mask], outputs=[z_mean, z_log_var], name="landmark_encoder")

    return model

def make_landmark_decoder():
    input_dim = 256 
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
    input_dim = 512 
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
    input_dim = 512 
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
#
# class FeatureEmbedding(tf.keras.Model):
#     def __init__(self):
#         self.extractor = make_extractor_model()
#         self.landmark_encoder = make_landmark_encoder()
#         self.landmark_decoder = make_landmark_decoder()
#         self.face_encoder = make_face_encoder()
#         self.face_mask_decoder = make_face_mask_decoder()
#         self.face_part_decoder = make_face_part_decoder()
#
#     def extractor_encode(self, img, training=False):
#         return self.extractor([img], training=training)
#
#     def encode(self, encoder, input, training=False):
#         return encoder(input, training=training)
#
#     def decode(self, decoder, input, training=False):
#         return decoder(input, training=training)
#
#     def reparametrize(self, z_mean, z_log_var):
#         eps = tf.random.normal(shape=z_mean.shape)
#         return eps * tf.exp(z_log_var * .5) + z_mean
#
#     def reparametrize_landmark(self, img_incomplete, z, mask, training=False):
#         l_mu, l_log_var  = self.encode(self.landmark_encoder, [img_incomplete, z, mask], training=training)
#         return self.reparametrize(l_mu, l_log_var)
#
#     def reparametrize_face(self, img_incomplete, z, mask, training=False):
#         f_mu, f_log_var  = self.encode(self.face_encoder, [img_incomplete, z, mask], training=training)
#         return self.reparametrize(f_mu, f_log_var)
#
#     def infer_landmark_z(self, img_incomplete, z, mask):
#         l_mu, l_log_var  = self.encode(self.landmark_encoder, [img_incomplete, z, mask], training=False)
#         return self.reparametrize(l_mu, l_log_var)
#
#     def infer_face_z(self, img_incomplete, z, mask):
#         f_mu, f_log_var  = self.encode(self.face_encoder, [img_incomplete, z, mask], training=False)
#         return self.reparametrize(f_mu, f_log_var)
#
#     def decode_landmark(self, z, training=False):
#         return self.decode(self.landmark_decoder, [ z ], training=training)
#
#     def decode_face_mask(self, z, training=False):
#         return self.decode(self.face_mask_decoder, [ z ], training=training)
#
#     def decode_face_part(self, z, training=False):
#         return self.decode(self.face_part_decoder, [ z ], training=training)
#
#     def call(self, inputs, training=False):
#         img_complete, img_incomplete, mask = inputs
#
        


class WGAN_GP():
    def __init__(self):
        self.generator = make_generator_model()
        self.discriminator = make_discriminator_model()
        self.local_discriminator = make_local_discriminator()
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.dglobal_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.dlocal_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        
    def gradient_penalty(self, real, fake):
        # tf.print("gp", tf.shape(real))
        # tf.print("gp ", tf.shape(fake))
        alpha = tf.random.uniform([real.shape[0], 1, 1, 1], 0, 1)
        diff = fake - real
        interpolated = real + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator([ interpolated ])
        
        grads = gp_tape.gradient(pred, interpolated)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp

    def gradient_penalty_local(self, real, fake, mask):
        # tf.print("gp", tf.shape(real))
        # tf.print("gp ", tf.shape(fake))
        alpha = tf.random.uniform([real.shape[0], 1, 1, 1], 0, 1)
        diff = fake - real
        interpolated = real + alpha * diff
        interpolated_masked = interpolated * mask
        one_channel_mask = mask[:,:,:,0:1]
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_masked)
            pred = self.local_discriminator([ interpolated_masked,  one_channel_mask ])
        
        grads = gp_tape.gradient(pred, interpolated_masked)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.)**2)
        return gp
        
    @tf.function
    def train_step(self, batch_original, batch_incomplete, masks, emb):
        one_channel_mask = masks[:,:,:,0:1]
        # Train Discriminator
        for _ in range(5):
            with tf.GradientTape() as disc_tape, \
                tf.GradientTape() as local_disc_tape:
                # Global discriminator
                generated_images = self.generator([ emb, batch_incomplete, one_channel_mask ])
                local_generated_images = generated_images * masks
                batch_original_masked = batch_original * masks
                
                real_output = self.discriminator([ batch_original ])
                fake_output = self.discriminator([ generated_images ])
                
                # tf.print(tf.shape(batch_original))
                # tf.print(tf.shape(generated_images))
                gp = self.gradient_penalty(batch_original, generated_images)
                disc_loss = tf.reduce_mean(fake_output) - tf.reduce_mean(real_output) + 10.0 * gp
                
                local_real_output = self.local_discriminator([batch_original_masked, one_channel_mask])
                local_fake_output = self.local_discriminator([local_generated_images, one_channel_mask])

                lgp = self.gradient_penalty_local(batch_original_masked, local_generated_images, masks)
                local_disc_loss = tf.reduce_mean(local_fake_output) - tf.reduce_mean(local_real_output) + 10.0 * lgp

                # tf.print("global loss", disc_loss)
                # tf.print("local loss", local_disc_loss)
                
            gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
            self.dglobal_optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

            local_gradients = local_disc_tape.gradient(local_disc_loss, self.local_discriminator.trainable_variables)
            self.dlocal_optimizer.apply_gradients(zip(local_gradients, self.local_discriminator.trainable_variables))
            
        # Train Generator
        with tf.GradientTape() as gen_tape:
            generated_images = self.generator([ emb, batch_incomplete, one_channel_mask ])
            local_generated_images = generated_images * masks
            fake_output = self.discriminator(generated_images)
            local_fake_output = self.local_discriminator([local_generated_images, one_channel_mask])
            gen_loss = -tf.reduce_mean(fake_output) - tf.reduce_mean(local_fake_output)
            
        # tf.print("gen loss", gen_loss)
            
        gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))
        
        return gen_loss, disc_loss, local_disc_loss, generated_images

