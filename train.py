### Setup
import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
from datetime import datetime

from IPython import display

from data import create_image_dataset
from model import make_extractor_model, make_generator_model, make_discriminator_model, \
    make_landmark_encoder, make_landmark_decoder, \
    make_face_encoder, make_face_mask_decoder, make_face_part_decoder, make_local_discriminator
from utils import generate_and_save_images, mask_rgb

import yaml

# Cargar configuración desde el archivo YAML
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Acceso a las rutas y configuraciones
original_img_dir = config["paths"]["original_img_dir"]
feature_img_dir = config["paths"]["feature_img_dir"]

# tf.config.run_functions_eagerly(True)

### Load and prepare the dataset

# Hyperparameters
batch_size = 12
LANDMARK_RECONSTRUCTION = 2000
FACE_MASK_RECONSTRUCTION = 1
FACE_PART_RECONSTRUCTION = 10000
DOMAIN_KL = 30
GLOBAL_DISCRIMINATOR_LOSS = 60
CONSISTENCY_LOSS = 1
EXTRACTOR_RECONSTRUCTION = 28
GEN_LOSS = 60
EXTRACTOR_KL = 30


train_dataset = create_image_dataset(original_img_dir, feature_img_dir, batch_size=batch_size)
print(train_dataset)

# # Visualizar
# plt.figure(figsize=(8,6))
# plt.imshow(create_inpainting_mask(), cmap='gray')
# plt.title('Máscara de Inpainting')
# plt.colorbar()
# plt.show()

# # normalization_layer = tf.keras.layers.Rescaling(1./255, offset=-1)
# # train_dataset = train_images.map(lambda x: normalization_layer(x))

# # Batch and shuffle the data
# #train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# train_dataset = dataset

# def show_images(train_dataset):
#   fig = plt.figure(figsize=(4, 4))
#   for x, i in  enumerate(train_dataset.take(4)):
#       plt.subplot(4, 4, x * 4+ 1)
#       plt.imshow((i[0,:,:,0:3] +1.) /2.)
#       plt.axis('off')
#       plt.subplot(4, 4, x * 4 + 2)
#       plt.imshow((i[0,:,:,3] +1.) /2., cmap='gray')
#       plt.axis('off')
#       plt.subplot(4, 4, x * 4 + 3)
#       plt.imshow((i[0,:,:,4] +1.) /2., cmap='gray')
#       plt.axis('off')
#       plt.subplot(4, 4, x * 4 + 4)
#       plt.imshow((i[0,:,:,5:8] +1.) /2.)
#       plt.axis('off')
#   plt.show()

# plt.show()

generator = make_generator_model()
# noise = tf.random.normal([batch_size, 128])
# for item in train_dataset.take(1):
#     # rgb_mask = mask_rgb(8)
#     imcomplete = item[:,:,:,0:3]
#     generated_image = generator([noise,imcomplete], training=False)

#     # plt.imshow((generated_image[0] + 1.) / 2.s
#     plt.imshow((imcomplete[0] + 1.) / 2.)
#     # plt.imshow(((generated_image[0] * (1. - rgb_mask)) + 1.) / 2.)
discriminator = make_discriminator_model()
local_discriminator = make_local_discriminator()
# decision = discriminator(generated_image)
# print (decision)
# This method returns a helper function to compute cross entropy loss
extractor = make_extractor_model()
landmark_encoder = make_landmark_encoder()
landmark_decoder = make_landmark_decoder()
face_encoder = make_face_encoder()
face_mask_decoder = make_face_mask_decoder()
face_part_decoder = make_face_part_decoder()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

### Discriminator loss

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

### Generator loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def l1_loss(y_true, y_pred):
  r_loss = tf.reduce_mean(tf.abs(y_true - y_pred), axis = [1,2,3])
  return tf.reduce_mean(r_loss)

def l1_loss_dim1(y_true, y_pred):
  r_loss = tf.reduce_mean(tf.abs(y_true - y_pred), axis = [1])
  return tf.reduce_mean(r_loss)

# for images -1 to 1 
def cross_entropy_img(y_true, y_pred):
  y_true = (y_true + 1.) / 2.
  y_pred = (y_pred + 1.) / 2.
  return cross_entropy(y_true, y_pred)

def sampling(z_mean, z_log_sigma):
  eps = tf.random.normal(shape=z_mean.shape)
  return eps * tf.exp(z_log_sigma * .5) + z_mean

def l1_reconstruction_loss(x, y_true):
  l1 = tf.reduce_sum(tf.abs(y_true - x), axis = [1,2,3])
  return tf.reduce_mean(l1)

def reconstruction_loss(x_logit, y_true, weight=tf.constant(7.)):
    cross_ent = tf.nn.weighted_cross_entropy_with_logits(
        logits=x_logit, 
        labels=y_true, 
        pos_weight=weight
    )
    imgs_loss = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    return tf.reduce_mean(imgs_loss)

def kl_divergence_loss(mean, logvar):
    kl_loss = -0.5 * (-tf.exp(logvar) - tf.square(mean) + 1. + logvar)
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))
    return kl_loss

# def compute_loss(model, x, y_true):
#     mean, logvar = model.encode(x)
#     z = model.reparameterize(mean, logvar)
#     x_logit = model.decode(z)
#
#     recon_loss = reconstruction_loss(x, x_logit, y_true)
#     kl_loss = kl_divergence_loss(z, mean, logvar)
#
#     return x_logit, -tf.reduce_mean(recon_loss + kl_loss)

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
global_discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
local_discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
face_embedding_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
# extractor_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

 
### Save checkpoints

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
                                # generator_optimizer=generator_optimizer,
                                #  extractor_optimizer=extractor_optimizer,
                                 global_discriminator_optimizer=global_discriminator_optimizer,
                                 face_embedding_optimizer=face_embedding_optimizer,
                                 local_discriminator_optimizer=local_discriminator_optimizer,
                                 extractor=extractor,
                                 generator=generator,
                                 discriminator=discriminator,
                                 local_discriminator=local_discriminator,
                                 landmark_encoder=landmark_encoder,
                                 landmark_decoder=landmark_decoder,
                                 face_encoder=face_encoder,
                                 face_mask_decoder=face_mask_decoder,
                                 face_part_decoder=face_part_decoder,
                                 )

## Define the training loop

def feature_embedding(x, z_extractor, mask):
  # Landkard encoder
  zl_mu, zl_log_var  = landmark_encoder([x, z_extractor, mask], training=True)
  landmark_sample = sampling(zl_mu, zl_log_var)
  landmark_reconstructed = landmark_decoder(landmark_sample, training=True)
  # Mask encoder
  zf_mu, zf_log_var  = face_encoder([x, z_extractor, mask], training=True)
  face_sample = sampling(zf_mu, zf_log_var)
  z_emb = tf.concat([landmark_sample, face_sample], axis=-1)
  # Face mask decoder
  face_mask_reconstructed = face_mask_decoder(z_emb, training=True)
  face_part_reconstructed = face_part_decoder(z_emb, training=True)

  return (landmark_sample, face_sample), z_emb, landmark_reconstructed, face_mask_reconstructed, face_part_reconstructed


EPOCHS = 50
noise_dim = 512
num_examples_to_generate = batch_size

# landmark_vae = LandmarkVAE(noise_dim, input_channels=3, output_channels=1)

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
mask_batch = mask_rgb(num_examples_to_generate)
z_seed = tf.random.normal([num_examples_to_generate, noise_dim])
for item in train_dataset.take(1):
  sample = item[:,:,:,0:3] * (1. - mask_batch)
seed = [z_seed, sample, mask_batch[:,:,:,0:1]]
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(batch):
    with \
      tf.GradientTape() as embedding_tape, \
      tf.GradientTape() as global_disc_tape, \
      tf.GradientTape() as local_discriminator_tape: \
      # tf.GradientTape() as gen_tape, \
      # tf.GradientTape() as extractor_tape:

      # Prepare the batch
      batch_size = tf.shape(batch)[0]
      lbatch_mask = mask_rgb(batch_size)
      tbatch_original = batch[:,:,:,0:3]
      tbatch_original_incomplete = tbatch_original * (1. - lbatch_mask)
      tbatch_landmarks = batch[:,:,:,3:4]
      tbatch_face_mask = batch[:,:,:,4:5]
      tbatch_face_part = batch[:,:,:,5:8]

      # Extractor 

      # z_random = tf.random.normal(shape = [batch_size, noise_dim])

      # _, zf_emb, zf_landmarks, zf_mask, zf_part = feature_embedding(tbatch_original_incomplete, z_random, lbatch_mask[:,:,:,0:1])
      # icf = generator([zf_emb, tbatch_original_incomplete, lbatch_mask[:,:,:,0:1]], training=True)

      # z_generated_mu, z_generated_log_var = extractor(tf.concat([icf, tf.tanh(zf_landmarks), tf.tanh(zf_mask), zf_part], axis=-1), training=True)
      # generated_sample = sampling(z_generated_mu, z_generated_log_var)

      # #Extractor loss
      # extractor_kl_loss = kl_divergence_loss(z_generated_mu, z_generated_log_var)
      # extractor_consistency_loss = l1_loss_dim1(generated_sample, z_random)

      # total_extractor_loss = extractor_kl_loss + extractor_consistency_loss

      # Face Embedding

      # e_mu, e_log_var = extractor(batch, training=True)
      # extractor_sample = sampling(e_mu, e_log_var)

      # (z1,z2), zr_emb, zr_landmarks, zr_mask, zr_part = feature_embedding(batch_original_incomplete, extractor_sample, mask_batch[:,:,:,0:1])
      extractor_sample = tf.random.normal(shape = [batch_size, noise_dim])

      one_channel_mask = mask_batch[:,:,:,0:1]
      # Landkard encoder
      zlr_mu, zlr_log_var  = landmark_encoder([tbatch_original_incomplete, extractor_sample, one_channel_mask], training=True)
      zr_l_sample = sampling(zlr_mu, zlr_log_var)
      icr_landmarks = landmark_decoder(zr_l_sample, training=True)
      # Mask encoder
      zfr_mu, zfr_log_var  = face_encoder([tbatch_original_incomplete, extractor_sample, one_channel_mask], training=True)
      zfr_f_sample = sampling(zfr_mu, zfr_log_var)
      z_emb = tf.concat([zr_l_sample, zfr_f_sample], axis=-1)
      # Face mask decoder
      icr_face_mask = face_mask_decoder(z_emb, training=True)
      icr_face_part = face_part_decoder(z_emb, training=True)

      landmark_reconstruction_loss = reconstruction_loss(icr_landmarks, (tbatch_landmarks + 1.) / 2., tf.constant(7.))
      face_mask_reconstruction_loss = 0.5 * reconstruction_loss(icr_face_mask, (tbatch_face_mask + 1.) / 2., tf.constant(3.5))
      face_part_reconstruction_loss = 0.5 * l1_reconstruction_loss(icr_face_part, tbatch_face_part)
      embedding_reconstruction_loss = landmark_reconstruction_loss +  face_mask_reconstruction_loss + face_part_reconstruction_loss

      zero_mask = tf.zeros_like(lbatch_mask[:,:,:,0:1])

      # Embedding consistency loss
      # _, zemb_no_mask, landmarks_no_mask, mask_no_mask, part_no_mask = feature_embedding(tbatch_original, extractor_sample, zero_mask)
      lnomask_mu, lnomask_log_var  = landmark_encoder([tbatch_original, extractor_sample, zero_mask], training=True)
      lnomask_sample = sampling(lnomask_mu, lnomask_log_var)
      fnomask_mu, fnomask_log_var  = face_encoder([tbatch_original, extractor_sample, zero_mask], training=True)
      fnomask_sample = sampling(fnomask_mu, fnomask_log_var)
      z_emb_no_mask = tf.concat([lnomask_sample, fnomask_sample], axis=-1)
      consistency_embedding_loss = l1_loss_dim1(z_emb, z_emb_no_mask)
      
      # Embedding kl loss
      z1_kl_loss = kl_divergence_loss(zlr_mu, zlr_log_var)
      z2_kl_loss = kl_divergence_loss(zfr_mu, zfr_log_var)

      total_embedding_loss = embedding_reconstruction_loss +  z1_kl_loss + z2_kl_loss  + consistency_embedding_loss

      # Generator loss

      icr = generator([z_emb, tbatch_original_incomplete, lbatch_mask[:,:,:,0:1]], training=True)
      gan_reconstruction_loss = tf.reduce_mean(l1_reconstruction_loss(icr, tbatch_original))

      # Consistency Loss
      # _, _, icr_landmarks, icr_mask, icr_part = feature_embedding(icr, extractor_sample, zero_mask)
      # icr_landmark_loss = 0.2 * l1_reconstruction_loss(tf.sigmoid( zr_landmarks),tf.sigmoid(icr_landmarks))
      # icr_face_mask_loss = 0.2 * l1_reconstruction_loss(tf.sigmoid(zr_mask), tf.sigmoid(icr_mask))
      # icr_face_part_loss = l1_reconstruction_loss(zr_part, icr_part)

      # icr_consistency_loss = icr_landmark_loss + icr_face_mask_loss + icr_face_part_loss


      # Discriminator loss

      # TODO: check if using tanh for discriminator is ok

      icr_combined = (icr * lbatch_mask) + (tbatch_original * (1. - lbatch_mask))

      real_output = discriminator(tbatch_original, training=True)  
      fake_output = discriminator(icr_combined, training=True)

      # Local discriminator loss
      masked_batch_original = tbatch_original * (lbatch_mask)
      masked_generated_images = icr * (lbatch_mask)

      local_real_output = local_discriminator([masked_batch_original, one_channel_mask], training=True)  
      local_fake_output = local_discriminator([masked_generated_images, one_channel_mask], training=True)

      global_discriminator_loss = discriminator_loss(real_output, fake_output)
      local_discriminator_loss = discriminator_loss(local_real_output, local_fake_output)

      local_generator_loss = generator_loss(local_fake_output)
      global_generator_loss = generator_loss(fake_output)

      total_generator_loss = gan_reconstruction_loss  + local_generator_loss + global_generator_loss # + icr_consistency_loss

      unified_loss = total_generator_loss + total_embedding_loss

      # # tf.print("Mask shape before processing:", tf.shape(batch_mask))

      # # Landkard encoder
      # zl_mu, zl_log_var  = landmark_encoder([batch_original_incomplete, extractor_sample, mask_batch[:,:,:,0:1]], training=True)
      # landmark_sample = sampling(zl_mu, zl_log_var)
      # landmark_reconstructed = landmark_decoder(landmark_sample, training=True)
      # # tf.print("Landmark shape:", tf.shape(landmark_reconstructed))

      # # Face encoder
      # zf_mu, zf_log_var  = face_encoder([batch_original_incomplete, extractor_sample, mask_batch[:,:,:,0:1]], training=True)
      # face_sample = sampling(zf_mu, zf_log_var)

      # z_emb = tf.concat([landmark_sample, face_sample], axis=-1)

      # face_mask_reconstructed = face_mask_decoder(z_emb, training=True)
      # face_part_reconstructed = face_part_decoder(z_emb, training=True)

      # # Generator
      # generated_images = generator([z_emb, batch_original_incomplete, batch_mask], training=True)
      # generated_images = ( generated_images * batch_mask) + batch_original * (1. - batch_mask)

      # # Discriminator
      # real_output = discriminator(batch_original, training=True)  
      # fake_output = discriminator(generated_images, training=True)

      # # Local discriminator
      # masked_batch_original = batch_original * (batch_mask)
      # masked_generated_images = generated_images * (batch_mask)

      # local_real_output = local_discriminator([masked_batch_original, batch_mask[:,:,:,0:1]], training=True)  
      # local_fake_output = local_discriminator([masked_generated_images, batch_mask[:,:,:,0:1]], training=True)

      # # Consitency loss for the generator
      # z = tf.random.normal(shape = [batch_size, noise_dim])
      # zlf_mu, zlf_log_var = landmark_encoder([batch_original_incomplete, z, batch_mask[:,:,:,0:1]], training=True)
      # zlf_sample = sampling(zlf_mu, zlf_log_var)
      # f_landmarks = landmark_decoder(zlf_sample, training=True)
      
      # zff_mu, zff_log_var = face_encoder([batch_original_incomplete, z, batch_mask[:,:,:,0:1]], training=True)
      # zff_sample = sampling(zff_mu, zff_log_var)
      # z_emb = tf.concat([zlf_sample, zff_sample], axis=-1)
      
      # f_face_mask = face_mask_decoder(z_emb, training=True)
      # f_face_part = face_part_decoder(z_emb, training=True)
      
      # generated_fake_images = generator([z_emb, batch_original_incomplete, batch_mask], training=True)
      
      # fft = tf.concat([generated_fake_images, f_landmarks, f_face_mask, f_face_part], axis=-1)
      
      # ef_mu, ef_log_var = extractor(fft, training=True)
      # z_fake_sample = sampling(ef_mu, ef_log_var)

      # # Local discriminator loss
      # local_discriminator_loss = discriminator_loss(local_real_output, local_fake_output)

      # # Gan loss
      # # consistency_loss = CONSISTENCY_LOSS * l1_loss_dim1(z_fake_sample, z)
      # disc_loss = discriminator_loss(real_output, fake_output)
      # y_true = (batch_original + 1.) / 2.
      # y_false = (generated_images + 1.) / 2.

      # gen_loss = generator_loss(fake_output) 
      # local_gen_loss = generator_loss(local_fake_output)
      # gen_reconstruction_loss = l1_reconstruction_loss(y_false, y_true)
      # total_loss = gen_loss + 0.1 * tf.reduce_mean(gen_reconstruction_loss) + local_gen_loss # + tf.reduce_mean(extractor_reconstruction_loss + extractor_kl_loss)  # + consistency_loss  

      # # Landmark loss -----------------------------------------------------------------------------------------------------
      # y_true_landmarks = (batch_landmarks + 1.) / 2.
      # # tf.print("landmark loss", temp_loss)
      # landmark_loss = reconstruction_loss(landmark_reconstructed , y_true_landmarks )
      # landmark_kl_loss = kl_divergence_loss(landmark_sample,zl_mu, zl_log_var)

      # zero_mask = tf.zeros_like(mask_batch)
      # zl_mu_c, zl_log_var_c  = landmark_encoder([batch_original, extractor_sample, zero_mask[:,:,:,0:1]], training=True)
      # landmark_sample_c = sampling(zl_mu_c, zl_log_var_c)

      # landmark_consistency_loss = 20 * l1_loss_dim1(landmark_sample, landmark_sample_c)
      # total_landmark_loss = tf.reduce_mean(landmark_loss ) +  landmark_kl_loss + landmark_consistency_loss

      # # Face loss --------------------------------------------------------------------------------------------------------
      # y_true_mask = (batch_face_mask + 1.) / 2.
      # face_mask_loss = reconstruction_loss(face_mask_reconstructed, y_true_mask, tf.constant(3.5))  
      # face_mask_kl_loss = kl_divergence_loss(face_sample, zf_mu, zf_log_var)
      # y_true_part = (batch_face_part + 1.) / 2.
      # reconstructed_part = (face_part_reconstructed + 1.) / 2.
      # face_part_loss = l1_reconstruction_loss(reconstructed_part, y_true_part) 

      # zf_mu_c, zf_log_var_c  = face_encoder([batch_original, extractor_sample, zero_mask[:,:,:,0:1]], training=True)
      # face_sample_c = sampling(zf_mu_c, zf_log_var_c)

      # face_consistency_loss = l1_loss_dim1(face_sample, face_sample_c)

      # total_face_loss = tf.reduce_mean(face_mask_loss +  face_part_loss ) + face_mask_kl_loss + face_consistency_loss

      # C

    # generator_trainable_variables = (
    #   generator.trainable_variables 
    #   # extractor.trainable_variables
    # )

    # face_trainable_variables = (
    #   face_encoder.trainable_variables + 
    #   face_mask_decoder.trainable_variables + 
    #   face_part_decoder.trainable_variables
    # )

    # landmark_trainable_variables = (
    #   landmark_encoder.trainable_variables + 
    #   landmark_decoder.trainable_variables 
    # )

    face_embedding_trainable_variables = (
      face_encoder.trainable_variables + 
      landmark_encoder.trainable_variables +
      face_mask_decoder.trainable_variables + 
      face_part_decoder.trainable_variables+ 
      landmark_decoder.trainable_variables 
      # add generator variables to one gradient update
      # generator.trainable_variables
    )

    gradients_of_embedding = embedding_tape.gradient(total_embedding_loss, face_embedding_trainable_variables)
    face_embedding_optimizer.apply_gradients(zip(gradients_of_embedding, face_embedding_trainable_variables))

    gradients_of_global_discriminator = global_disc_tape.gradient(global_discriminator_loss, discriminator.trainable_variables)
    global_discriminator_optimizer.apply_gradients(zip(gradients_of_global_discriminator, discriminator.trainable_variables))

    gradients_of_local_discriminator = local_discriminator_tape.gradient(local_discriminator_loss, local_discriminator.trainable_variables)
    local_discriminator_optimizer.apply_gradients(zip(gradients_of_local_discriminator, local_discriminator.trainable_variables))

    # gradients_of_generator = gen_tape.gradient(total_generator_loss, generator.trainable_variables)
    # generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    # gradients_of_extractor = extractor_tape.gradient(total_extractor_loss, extractor.trainable_variables)
    # extractor_optimizer.apply_gradients(zip(gradients_of_extractor, extractor.trainable_variables))


    # gradients_of_landmark = landmark_tape.gradient(total_landmark_loss, landmark_trainable_variables)
    # landmark_optimizer.apply_gradients(zip(gradients_of_landmark, landmark_trainable_variables))

    # gradients_of_face = face_tape.gradient(total_face_loss, face_trainable_variables)
    # face_optimizer.apply_gradients(zip(gradients_of_face, face_trainable_variables))
    
    # gradients_of_generator = gen_tape.gradient(total_loss, generator_trainable_variables)
    # generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_trainable_variables))

    # gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # gradients_of_local_discriminator = local_discriminator_tape.gradient(local_discriminator_loss, local_discriminator.trainable_variables)
    # local_discriminator_optimizer.apply_gradients(zip(gradients_of_local_discriminator, local_discriminator.trainable_variables))

    # with tf.GradientTape() as extractor_tape:
    #   # Prepare the batch
    #   batch_size = tf.shape(batch)[0]
    #   batch_mask = mask_rgb(batch_size)
    #   batch_original = batch[:,:,:,0:3]
    #   batch_original_incomplete = batch_original * (1. - batch_mask)
    #   batch_landmarks = batch[:,:,:,3:4]
    #   batch_face_mask = batch[:,:,:,4:5]
    #   batch_face_part = batch[:,:,:,5:8]

    #   # Extractor 
    #   e_mu, e_log_var = extractor(batch, training=True)
    #   extractor_sample = sampling(e_mu, e_log_var)

    #   # Random sample
    #   z = tf.random.normal(shape = [batch_size, noise_dim])
    #   zlf_mu, zlf_log_var = landmark_encoder([batch_original_incomplete, z, batch_mask[:,:,:,0:1]], training=False)
    #   zlf_sample = sampling(zlf_mu, zlf_log_var)
    #   f_landmarks = landmark_decoder(zlf_sample, training=False)
      
    #   zff_mu, zff_log_var = face_encoder([batch_original_incomplete, z, batch_mask[:,:,:,0:1]], training=False)
    #   zff_sample = sampling(zff_mu, zff_log_var)
    #   z_emb = tf.concat([zlf_sample, zff_sample], axis=-1)
      
    #   f_face_mask = face_mask_decoder(z_emb, training=False)
    #   f_face_part = face_part_decoder(z_emb, training=False)
      
    #   generated_fake_images = generator([z_emb, batch_original_incomplete, batch_mask], training=False)
      
    #   fft = tf.concat([generated_fake_images, f_landmarks, f_face_mask, f_face_part], axis=-1)
      
    #   ef_mu, ef_log_var = extractor(fft, training=False)
    #   z_fake_sample = sampling(ef_mu, ef_log_var)

    #   extractor_kl_loss = EXTRACTOR_KL * kl_divergence_loss(extractor_sample, e_mu, e_log_var)
    #   extractor_l1_loss = l1_loss_dim1(z_fake_sample, z)

    #   total_extractor_loss = (extractor_kl_loss) + extractor_l1_loss
    
    # gradients_of_extractor = extractor_tape.gradient(total_extractor_loss, extractor.trainable_variables)
    # extractor_optimizer.apply_gradients(zip(gradients_of_extractor, extractor.trainable_variables))

    return {
      "outputs": {
        "original_images": tbatch_original,
        "reconstructed_images": icr,
        "landmark_reconstructed": icr_landmarks,
        "face_mask_reconstructed": icr_face_mask,
        "face_part_reconstructed": icr_face_part,
      },
      "losses": {	
        "total/total_generator_loss": total_generator_loss,
        "total/total_embedding_loss": total_embedding_loss,
        # "total/total_extractor_loss": total_extractor_loss,
        "total/global_disc_loss": global_discriminator_loss,
        "total/local_disc_loss": local_discriminator_loss,
        # "extractor/kl_loss": (extractor_kl_loss),
        # "extractor/consistency_loss": extractor_consistency_loss,
        "embedding/landmark_reconstruction_loss": (landmark_reconstruction_loss),
        "embedding/face_mask_reconstruction_loss": (face_mask_reconstruction_loss),
        "embedding/face_part_reconstruction_loss": (face_part_reconstruction_loss),
        "embedding/z1_kl_loss": (z1_kl_loss),
        "embedding/z2_kl_loss": (z2_kl_loss),
        # "embedding/consistency_loss": (consistency_embedding_loss),
        "generator/reconstruction_loss": (gan_reconstruction_loss),
        # "generator/landmarks_consistency_loss": (icr_landmark_loss),
        # "generator/face_mask_consistency_loss": (icr_face_mask_loss),
        # "generator/face_part_consistency_loss": (icr_face_part_loss),
        "generator/local_loss": (local_generator_loss),
        "generator/global_loss": (global_generator_loss),
        # "extractor_reconstruction_loss": tf.reduce_mean(extractor_reconstruction_loss),
        # "extractor_kl_loss": tf.reduce_mean(extractor_kl_loss),
        # "landmark_reconstruction_loss": tf.reduce_mean(landmark_loss),
        # "face_mask_reconstruction_loss": tf.reduce_mean(face_mask_loss),
        # "face_part_reconstruction_loss": tf.reduce_mean(face_part_loss),
        # "landkmark_kl_loss": tf.reduce_mean(landmark_kl_loss),
        # "face_mask_kl_loss": tf.reduce_mean(face_mask_kl_loss),
        # "landmark_consistency_loss": (landmark_consistency_loss),
        # "face_consistency_loss": (face_consistency_loss),
        # "consistency_loss": consistency_loss,
        # "/extractor/kl_loss": (extractor_kl_loss),
        # "/extractor/l1_loss": (extractor_l1_loss),
      }
    }

@tf.function
def train_embedding_step(batch):
  with \
    tf.GradientTape() as embedding_tape:
      # Prepare the batch
      batch_size = tf.shape(batch)[0]
      lbatch_mask = mask_rgb(batch_size)
      tbatch_original = batch[:,:,:,0:3]
      tbatch_original_incomplete = tbatch_original * (1. - lbatch_mask)
      tbatch_landmarks = batch[:,:,:,3:4]
      tbatch_face_mask = batch[:,:,:,4:5]
      tbatch_face_part = batch[:,:,:,5:8]

      e_mean, e_log_var = extractor(batch, training=True)
      extractor_sample = sampling(e_mean, e_log_var)

      tf.debugging.check_numerics(e_mean, "e_mean")
      tf.debugging.check_numerics(e_log_var, "e_log_var")
      tf.debugging.check_numerics(extractor_sample, "extractor_sample")

      one_channel_mask = mask_batch[:,:,:,0:1]
      # Landkard encoder
      zlr_mu, zlr_log_var  = landmark_encoder([tbatch_original_incomplete, extractor_sample, one_channel_mask], training=True)
      zr_l_sample = sampling(zlr_mu, zlr_log_var)
      icr_landmarks = landmark_decoder(zr_l_sample, training=True)
      # Mask encoder
      zfr_mu, zfr_log_var  = face_encoder([tbatch_original_incomplete, extractor_sample, one_channel_mask], training=True)
      zfr_f_sample = sampling(zfr_mu, zfr_log_var)
      z_emb = tf.concat([zr_l_sample, zfr_f_sample], axis=-1)
      # Face mask decoder
      icr_face_mask = face_mask_decoder(z_emb, training=True)
      icr_face_part = face_part_decoder(z_emb, training=True)

      tf.debugging.check_numerics(zlr_mu, "zlr_mu")
      tf.debugging.check_numerics(zlr_log_var, "zlr_log_var")
      tf.debugging.check_numerics(zfr_mu, "zfr_mu")
      tf.debugging.check_numerics(zfr_log_var, "zfr_log_var")
      tf.debugging.check_numerics(zr_l_sample, "zr_l_sample")
      tf.debugging.check_numerics(zfr_f_sample, "zfr_f_sample")
      tf.debugging.check_numerics(z_emb, "z_emb")
      tf.debugging.check_numerics(icr_face_part, "icr_face_part")
      tf.debugging.check_numerics(icr_face_mask, "icr_face_mask")
      tf.debugging.check_numerics(icr_landmarks, "icr_landmarks")

      landmark_reconstruction_loss = 2 * reconstruction_loss(icr_landmarks, (tbatch_landmarks + 1.) / 2., tf.constant(7.))
      face_mask_reconstruction_loss = 2 * reconstruction_loss(icr_face_mask, (tbatch_face_mask + 1.) / 2., tf.constant(2.5))
      face_part_reconstruction_loss = l1_reconstruction_loss(icr_face_part, tbatch_face_part)
      embedding_reconstruction_loss = landmark_reconstruction_loss +  face_mask_reconstruction_loss + face_part_reconstruction_loss

      zero_mask = tf.zeros_like(lbatch_mask[:,:,:,0:1])

      # Embedding consistency loss
      # _, zemb_no_mask, landmarks_no_mask, mask_no_mask, part_no_mask = feature_embedding(tbatch_original, extractor_sample, zero_mask)
      lnomask_mu, lnomask_log_var  = landmark_encoder([tbatch_original, extractor_sample, zero_mask], training=False)
      lnomask_sample = sampling(lnomask_mu, lnomask_log_var)
      fnomask_mu, fnomask_log_var  = face_encoder([tbatch_original, extractor_sample, zero_mask], training=False)
      fnomask_sample = sampling(fnomask_mu, fnomask_log_var)
      z_emb_no_mask = tf.concat([lnomask_sample, fnomask_sample], axis=-1)
      consistency_embedding_loss = l1_loss_dim1(z_emb, z_emb_no_mask)
      
      # Embedding kl loss
      z1_kl_loss = kl_divergence_loss(zlr_mu, zlr_log_var)
      z2_kl_loss = kl_divergence_loss(zfr_mu, zfr_log_var)

      extractor_kl_loss = kl_divergence_loss(e_mean, e_log_var)
      total_embedding_loss = embedding_reconstruction_loss +  z1_kl_loss + z2_kl_loss  + consistency_embedding_loss + extractor_kl_loss

  face_embedding_trainable_variables = (
      extractor.trainable_variables +
      face_encoder.trainable_variables + 
      landmark_encoder.trainable_variables +
      face_mask_decoder.trainable_variables + 
      face_part_decoder.trainable_variables+ 
      landmark_decoder.trainable_variables 
  )

  gradients_of_embedding = embedding_tape.gradient(total_embedding_loss, face_embedding_trainable_variables)
  face_embedding_optimizer.apply_gradients(zip(gradients_of_embedding, face_embedding_trainable_variables))

  return {
      "outputs": {
        "original_images": tbatch_original,
        "landmark_reconstructed": icr_landmarks,
        "face_mask_reconstructed": icr_face_mask,
        "face_part_reconstructed": icr_face_part,
      },
      "losses": {	
        "total/total_embedding_loss": total_embedding_loss,
        "extractor/extractor_kl_loss": extractor_kl_loss,
        "embedding/landmark_reconstruction_loss": (landmark_reconstruction_loss),
        "embedding/face_mask_reconstruction_loss": (face_mask_reconstruction_loss),
        "embedding/face_part_reconstruction_loss": (face_part_reconstruction_loss),
        "embedding/z1_kl_loss": (z1_kl_loss),
        "embedding/z2_kl_loss": (z2_kl_loss),
        "embedding/consistency_loss": (consistency_embedding_loss),
      }
  }


def train_all(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for step, image_batch in enumerate(dataset):
      values = train_step(image_batch)

    tf.print("losses", values["losses"])	
    # Registrar métricas en TensorBoard
    with writer.as_default():
      for name, value in values["losses"].items():
        tf.summary.scalar(name, value, step=epoch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)

    # tf.print("losses", values["losses"])	
    generate_and_save_images(generator,
                          epoch = epoch + 1,
                          args = seed)

    if (epoch + 1) % 4 == 0 and not False:
      outputs = values["outputs"]
      original_image = outputs["original_images"][0]
      landmark_sample = outputs["landmark_reconstructed"][0]
      mask_sample = outputs["face_mask_reconstructed"][0]
      face_part_samle = outputs["face_part_reconstructed"][0]
      fig = plt.figure(figsize=(6, 6))
      plt.subplot(1, 4, 1)
      plt.imshow(tf.sigmoid(landmark_sample) , cmap='gray')
      plt.axis('off')
      plt.subplot(1, 4, 2)
      plt.imshow(tf.sigmoid(mask_sample), cmap='gray')
      plt.axis('off')
      plt.subplot(1, 4, 3)
      plt.imshow((face_part_samle + 1.) / 2.)
      plt.axis('off')
      plt.subplot(1, 4, 4)
      plt.imshow((original_image + 1.) / 2.)
      plt.axis('off')
      plt.show()

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epoch = epochs,
                           args =seed)

def train_embedding(dataset, epochs):
  total_steps = 0
  for epoch in range(epochs):
    start = time.time()

    for step, image_batch in enumerate(dataset):
      tf.print("Epoch {}, step {}".format(epoch, step))
      values = train_embedding_step(image_batch)
      total_steps += 1


      if total_steps % 100 == 0:
        tf.print("losses", values["losses"])	
        # Registrar métricas en TensorBoard
        with writer.as_default():
          for name, value in values["losses"].items():
            tf.summary.scalar(name, value, step=total_steps)
        # Save the model
        checkpoint.save(file_prefix = checkpoint_prefix)

      #   display.clear_output(wait=True)
      #   outputs = values["outputs"]
      #   original_image = outputs["original_images"][0]
      #   landmark_sample = outputs["landmark_reconstructed"][0]
      #   mask_sample = outputs["face_mask_reconstructed"][0]
      #   face_part_samle = outputs["face_part_reconstructed"][0]
      #   fig = plt.figure(figsize=(6, 6))
      #   plt.subplot(1, 4, 1)
      #   plt.imshow(tf.sigmoid(landmark_sample) , cmap='gray')
      #   plt.axis('off')
      #   plt.subplot(1, 4, 2)
      #   plt.imshow(tf.sigmoid(mask_sample), cmap='gray')
      #   plt.axis('off')
      #   plt.subplot(1, 4, 3)
      #   plt.imshow((face_part_samle + 1.) / 2.)
      #   plt.axis('off')
      #   plt.subplot(1, 4, 4)
      #   plt.imshow((original_image + 1.) / 2.)
      #   plt.axis('off')
      #   plt.show()

      # if total_steps % 400 == 0:
      #   # sleep 5 min
      #   time.sleep(300)

        # If any loss is NaN or infinite, stop the training
        # if any(np.isnan(values["losses"].values())):
        #   print("NaN detected, stopping training")
        #   break

    # tf.print("losses", values["losses"])	
    # Registrar métricas en TensorBoard
    # with writer.as_default():
    #   for name, value in values["losses"].items():
    #     tf.summary.scalar(name, value, step=epoch)
    # Produce images for the GIF as you go
    # generate_and_save_images(generator,
    #                       epoch = epoch + 1,
    #                       args = seed)
    


def train(dataset, epochs):
  # train_all(dataset, epochs)
  train_embedding(dataset, epochs)

## Train the model
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(train_dataset, EPOCHS)

# # Display a single image using the epoch number
# def display_image(epoch_no):
#   return PIL.Image.open('res/image_at_epoch_{:04d}.png'.format(EPOCHS))

# display_image(15)
