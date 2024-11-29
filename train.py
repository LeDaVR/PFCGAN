### Setup
import tensorflow as tf
from tensorflow.keras import layers

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time

from IPython import display

from data import create_image_dataset
from model import make_extractor_model, make_generator_model, make_discriminator_model, \
    make_landmark_encoder, make_landmark_decoder, \
    make_face_encoder, make_face_mask_decoder, make_face_part_decoder
from utils import generate_and_save_images, mask_rgb

tf.config.run_functions_eagerly(True)

### Load and prepare the dataset
batch_size = 8
original_img_dir = 'D:/My Files/UNSA/PFCIII/prepro/test/original'
feature_img_dir = 'D:/My Files/UNSA/PFCIII/prepro/test/processed'
train_dataset = create_image_dataset(original_img_dir, feature_img_dir, batch_size=batch_size)
print(train_dataset)

# Hyperparameters
LANDMARK_RECONSTRUCTION = 1000
FACE_MASK_RECONSTRUCTION = 2000
FACE_PART_RECONSTRUCTION = 2000
DOMAIN_KL = 30
GLOBAL_DISCRIMINATOR_LOSS = 30
CONSISTENCY_LOSS = 1
EXTRACTOR_RECONSTRUCTION = 28
EXTRACTOR_KL = 30


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

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def l1_loss(y_true, y_pred):
  r_loss = tf.reduce_mean(tf.abs(y_true - y_pred), axis = [1,2,3])
  return tf.reduce_mean(r_loss)

def l1_loss_dim1(y_true, y_pred):
  r_loss = tf.reduce_mean(tf.abs(y_true - y_pred), axis = [1])
  return tf.reduce_mean(r_loss)

def cross_entropy_loss(y_true, y_pred):
  return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))

def sampling(z_mean, z_log_sigma):
  eps = tf.random.normal(shape = tf.shape(input=z_mean))
  std = z_mean + tf.exp(z_log_sigma / 2) * eps

  z = tf.add(z_mean, tf.multiply(std, eps))
  return z

def kl_loss(mean, log_var):
  kl_loss =  -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis = 1)
  return tf.reduce_mean(kl_loss)

def extractor_loss(y_true, y_pred, mean, log_var):
  mse = l1_loss(y_true, y_pred)
  kl = kl_loss(mean, log_var)
  return mse + kl

def vae_loss(y_true, y_pred, mean, log_var):
  mse = l1_loss(y_true, y_pred)
  kl = kl_loss(mean, log_var)
  return mse + kl


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
domain_optimizer = tf.keras.optimizers.Adam(1e-4)
 
### Save checkpoints

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

## Define the training loop


EPOCHS = 50
noise_dim = 512
num_examples_to_generate = 8

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
mask_batch = mask_rgb(num_examples_to_generate)
z_seed = tf.random.normal([num_examples_to_generate, noise_dim])
for item in train_dataset.take(1):
  sample = item[:,:,:,0:3] * (1. - mask_batch)
seed = [z_seed, sample, mask_batch]

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(batch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as domain_tape:

      # Prepare the batch
      batch_size = tf.shape(batch)[0]
      batch_mask = mask_rgb(batch_size)
      batch_original = batch[:,:,:,0:3]
      batch_original_incomplete = batch_original * (1. - batch_mask)
      batch_landmarks = batch[:,:,:,3:4]
      batch_face_mask = batch[:,:,:,4:5]
      batch_face_part = batch[:,:,:,5:8]

      # Extractor 
      e_mu, e_log_var = extractor(batch, training=True)
      extractor_sample = sampling(e_mu, e_log_var)
      # tf.print("Mask shape before processing:", tf.shape(batch_mask))

      # Landkard encoder
      zl_mu, zl_log_var  = landmark_encoder([batch_original_incomplete, extractor_sample], training=True)
      landmark_sample = sampling(zl_mu, zl_log_var)
      landmark_reconstructed = landmark_decoder(landmark_sample, training=True)

      # Face encoder
      zf_mu, zf_log_var  = face_encoder([batch_original_incomplete, extractor_sample], training=True)
      face_sample = sampling(zf_mu, zf_log_var)

      z_emb = tf.concat([landmark_sample, face_sample], axis=-1)

      face_mask_reconstructed = face_mask_decoder(z_emb, training=True)
      face_part_reconstructed = face_part_decoder(z_emb, training=True)

      # Generator
      generated_images = generator([z_emb, batch_original_incomplete, batch_mask], training=True)
      generated_images = ( generated_images * batch_mask) + batch_original * (1. - batch_mask)

      real_output = discriminator(batch_original, training=True)  
      fake_output = discriminator(generated_images, training=True)

      # Consitency loss for the generator
      z = tf.random.normal(shape = [batch_size, noise_dim])
      zlf_mu, zlf_log_var = landmark_encoder([batch_original_incomplete, z], training=True)
      zlf_sample = sampling(zlf_mu, zlf_log_var)
      f_landmarks = landmark_decoder(zlf_sample, training=True)

      zff_mu, zff_log_var = face_encoder([batch_original_incomplete, z], training=True)
      zff_sample = sampling(zff_mu, zff_log_var)
      z_emb = tf.concat([zlf_sample, zff_sample], axis=-1)

      f_face_mask = face_mask_decoder(z_emb, training=True)
      f_face_part = face_part_decoder(z_emb, training=True)

      generated_fake_images = generator([z_emb, batch_original_incomplete, batch_mask], training=True)

      fft = tf.concat([generated_fake_images, f_landmarks, f_face_mask, f_face_part], axis=-1)

      ef_mu, ef_log_var = extractor(fft, training=True)
      z_fake_sample = sampling(ef_mu, ef_log_var)

      consistency_loss = CONSISTENCY_LOSS * l1_loss_dim1(z_fake_sample, z)
      disc_loss = GLOBAL_DISCRIMINATOR_LOSS * discriminator_loss(real_output, fake_output)
      ext_loss =  EXTRACTOR_RECONSTRUCTION * l1_loss(batch_original, generated_images) + EXTRACTOR_KL * kl_loss(e_mu, e_log_var)
      landmark_loss = LANDMARK_RECONSTRUCTION * l1_loss(batch_landmarks, landmark_reconstructed) + DOMAIN_KL * kl_loss(zl_mu, zl_log_var)
      face_mask_loss = FACE_MASK_RECONSTRUCTION * l1_loss(batch_face_mask, face_mask_reconstructed) + DOMAIN_KL * kl_loss(zf_mu, zf_log_var)
      face_part_loss = FACE_PART_RECONSTRUCTION * l1_loss(batch_face_part, face_part_reconstructed) 
      gen_loss = generator_loss(fake_output) 

      total_loss = (
        gen_loss + ext_loss + consistency_loss
      )

      total_domain_loss = landmark_loss + face_mask_loss + face_part_loss

    generator_trainable_variables = (
      generator.trainable_variables +
      extractor.trainable_variables 
    )

    domain_trainable_variables = (
      landmark_encoder.trainable_variables + 
      landmark_decoder.trainable_variables + 
      face_encoder.trainable_variables + 
      face_mask_decoder.trainable_variables + 
      face_part_decoder.trainable_variables
    )
    # print("losses", total_loss, ext_loss, gen_loss)
    
    gradients_of_generator = gen_tape.gradient(total_loss, generator_trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    gradients_of_domain = domain_tape.gradient(total_domain_loss, domain_trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    domain_optimizer.apply_gradients(zip(gradients_of_domain, domain_trainable_variables))

    return {
      "total_loss": total_loss,
      "gen_loss": gen_loss,
      "disc_loss": disc_loss,
      "ext_loss": ext_loss,
      "landmark_loss": landmark_loss,
      "face_mask_loss": face_mask_loss,
      "face_part_loss": face_part_loss,
      "consistency_loss": consistency_loss,
      "total_domain_loss": total_domain_loss,
    }

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
    gen_loss = disc_loss = ext_loss = 0

    for step, image_batch in enumerate(dataset):
      losses = train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)

    tf.print("losses", losses)
    generate_and_save_images(generator,
                          epoch = epoch + 1,
                          args = seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epoch = epochs,
                           args =seed)


## Train the model
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(train_dataset, EPOCHS)

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('res/image_at_epoch_{:04d}.png'.format(EPOCHS))

display_image(15)

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('/res/image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)
