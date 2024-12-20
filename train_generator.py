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
EPOCHS = config["hyper_parameters"]["epochs"]
batch_size = config["hyper_parameters"]["batch_size"]
w_landmarks = 0.3
w_face_mask = 0.2
w_face_part = 0.5
consistency_loss = 0.1
adversarial_loss = 20
rec_loss = 0.9



train_dataset = create_image_dataset(original_img_dir, feature_img_dir, batch_size=batch_size)
print(train_dataset)

generator = make_generator_model()
discriminator = make_discriminator_model()
local_discriminator = make_local_discriminator()
extractor = make_extractor_model()
landmark_encoder = make_landmark_encoder()
landmark_decoder = make_landmark_decoder()
face_encoder = make_face_encoder()
face_mask_decoder = make_face_mask_decoder()
face_part_decoder = make_face_part_decoder()

class PFCGAN():
   def __init__(self, landmark_encoder, landmark_decoder, face_encoder, face_mask_decoder, face_part_decoder, generator):
      self.landmark_encoder = landmark_encoder
      self.landmark_decoder = landmark_decoder
      self.face_encoder = face_encoder
      self.face_mask_decoder = face_mask_decoder
      self.face_part_decoder = face_part_decoder
      self.generator = generator

pfcGan = PFCGAN(landmark_encoder, landmark_decoder, face_encoder, face_mask_decoder, face_part_decoder, generator)

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

def l1_loss_dim1(y_true, y_pred):
  r_loss = tf.reduce_mean(tf.abs(y_true - y_pred), axis = [1])
  return tf.reduce_mean(r_loss)

def reparametrize(z_mean, z_log_sigma):
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
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
    kl_loss = tf.reduce_mean(kl_loss)
    return kl_loss

def masked_loss(y_true, y_pred, mask):
   l1 = (tf.abs(y_true - y_pred))
   masked_l1 = l1 * mask
   total_img_error = tf.reduce_sum(masked_l1, axis=[1,2,3])
  #  num_pixels_per_image = tf.reduce_sum(mask, axis=[1, 2, 3])  
  #  normalized_img_error = total_img_error / (num_pixels_per_image + 1e-8)
   return tf.reduce_mean(total_img_error)

def masked_weighted_cross_entropy_loss(x_logits, y_true, mask, weight=tf.constant(7.)):
    cross_entropy_loss = tf.nn.weighted_cross_entropy_with_logits(logits=x_logits, labels=y_true, pos_weight=weight)
    masked_cross_entropy_loss = cross_entropy_loss * mask
    total_img_error = tf.reduce_sum(masked_cross_entropy_loss, axis=[1,2,3])
    # num_pixels_per_image = tf.reduce_sum(mask, axis=[1, 2, 3])
    # normalized_img_error = total_img_error / (num_pixels_per_image + 1e-8)
    return tf.reduce_mean(total_img_error)

# def compute_loss(model, x, y_true):
#     mean, logvar = model.encode(x)
#     z = model.reparameterize(mean, logvar)
#     x_logit = model.decode(z)
#
#     recon_loss = reconstruction_loss(x, x_logit, y_true)
#     kl_loss = kl_divergence_loss(z, mean, logvar)
#
#     return x_logit, -tf.reduce_mean(recon_loss + kl_loss)

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
global_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
local_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
face_embedding_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
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
                                 generator_optimizer=generator_optimizer,
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

def feature_embedding(x, z_extractor, mask, training=True):
  # Landkard encoder
  zl_mu, zl_log_var  = landmark_encoder([x, z_extractor, mask], training=training)
  landmark_sample = reparametrize(zl_mu, zl_log_var)
  landmark_reconstructed = landmark_decoder(landmark_sample, training=training)
  # Mask encoder
  zf_mu, zf_log_var  = face_encoder([x, z_extractor, mask], training=training)
  face_sample = reparametrize(zf_mu, zf_log_var)
  z_emb = tf.concat([landmark_sample, face_sample], axis=-1)
  # Face mask decoder
  face_mask_reconstructed = face_mask_decoder(z_emb, training=training)
  face_part_reconstructed = face_part_decoder(z_emb, training=training)

  return (landmark_sample, face_sample), z_emb, landmark_reconstructed, face_mask_reconstructed, face_part_reconstructed


def inference(lencoder, ldecoder, fencoder, mdecoder, fdecoder, generator, z, batch_incomplete, batch_mask):
    l_mu, l_log_var = lencoder([batch_incomplete, z, batch_mask], training=False)
    reparametrized_landmarks = reparametrize(l_mu, l_log_var)
    f_mu, f_log_var = fencoder([batch_incomplete, z, batch_mask], training=False)
    reparametrized_face= reparametrize(f_mu, f_log_var)
    emb = tf.concat([reparametrized_landmarks, reparametrized_face], axis=-1)
    fake = generator([emb, batch_incomplete, batch_mask], training=False)
    return fake

noise_dim = 512
num_examples_to_generate = batch_size

# landmark_vae = LandmarkVAE(noise_dim, input_channels=3, output_channels=1)

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(batch, lbatch_mask):
    with tf.GradientTape() as global_disc_tape, \
      tf.GradientTape() as embedding_tape, \
      tf.GradientTape() as local_discriminator_tape,\
      tf.GradientTape() as generator_tape:

      # Prepare the batch
      batch_size = tf.shape(batch)[0]
      # lbatch_mask = mask_rgb(batch_size)
      tbatch_original = batch[:,:,:,0:3]
      tbatch_original_incomplete = tbatch_original * (1. - lbatch_mask)
      tbatch_landmarks = batch[:,:,:,3:4]
      tbatch_face_mask = batch[:,:,:,4:5]
      tbatch_face_part = batch[:,:,:,5:8]

      one_channel_mask = lbatch_mask[:,:,:,0:1]

      # Extractor 
      z_random = tf.random.normal(shape = [batch_size, noise_dim])
      _, f_emb, f_landmarks, f_mask, f_face_part  = feature_embedding(tbatch_original_incomplete, z_random , one_channel_mask, training=False)
      icf = generator([f_emb, tbatch_original_incomplete, one_channel_mask], training=True)

      f_rec_loss = rec_loss * masked_loss(tbatch_original, icf, 1-lbatch_mask)

      # Face Embedding
      e_mu, e_log_var = extractor(tf.concat([tbatch_landmarks, tbatch_face_mask, tbatch_face_part], axis=-1), training=False)
      extractor_sample = reparametrize(e_mu, e_log_var)

      # Landkard encoder
      zlr_mu, zlr_log_var  = landmark_encoder([tbatch_original_incomplete, extractor_sample, one_channel_mask], training=False)
      zr_l_sample = reparametrize(zlr_mu, zlr_log_var)
      icr_landmarks = landmark_decoder(zr_l_sample, training=False)
      # Mask encoder
      zfr_mu, zfr_log_var  = face_encoder([tbatch_original_incomplete, extractor_sample, one_channel_mask], training=False)
      zfr_f_sample = reparametrize(zfr_mu, zfr_log_var)
      z_emb = tf.concat([zr_l_sample, zfr_f_sample], axis=-1)
      # Face mask decoder
      icr_face_mask = face_mask_decoder(z_emb, training=False)
      icr_face_part = face_part_decoder(z_emb, training=False)

      icr = generator([z_emb, tbatch_original_incomplete, one_channel_mask], training=True)

      zero_mask = tf.zeros_like(lbatch_mask[:,:,:,0:1])
      # Generator loss
      # gan_reconstruction_loss = tf.reduce_mean(l1_reconstruction_loss(icr, tbatch_original))
      gan_reconstruction_loss = rec_loss * masked_loss(tbatch_original, icr, lbatch_mask)

      icr_combined = (icr * lbatch_mask) + (tbatch_original * (1. - lbatch_mask))

      # Consistency Loss
      _, _, nicr_landmarks, nicr_mask, nicr_part = feature_embedding(icr_combined, extractor_sample, zero_mask, training=False)
      icr_landmark_loss = w_landmarks * masked_loss(tf.sigmoid( nicr_landmarks),tf.sigmoid(icr_landmarks), lbatch_mask)
      icr_face_mask_loss = w_face_mask * masked_loss(tf.sigmoid(nicr_mask), tf.sigmoid(icr_face_mask), lbatch_mask)
      # icr_face_part_loss = l1_reconstruction_loss(nicr_part, icr_face_part)

      # icr_consistency_loss = 0.5 * (icr_landmark_loss + icr_face_mask_loss + icr_face_part_loss)
      icr_consistency_loss = icr_landmark_loss + icr_face_mask_loss 

      # Discriminator loss

      # TODO: check if using tanh for discriminator is ok


      real_output =  discriminator(tbatch_original, training=True)  
      fake_output =  discriminator(icr_combined, training=True)

      # Local discriminator loss
      masked_batch_original = tbatch_original * (lbatch_mask)
      masked_generated_images = icr * (lbatch_mask)

      local_real_output = local_discriminator([masked_batch_original, one_channel_mask], training=True)  
      local_fake_output = local_discriminator([masked_generated_images, one_channel_mask], training=True)

      global_discriminator_loss = adversarial_loss * discriminator_loss(real_output, fake_output)
      local_discriminator_loss = adversarial_loss * discriminator_loss(local_real_output, local_fake_output)

      local_generator_loss = adversarial_loss * generator_loss(local_fake_output)
      global_generator_loss = adversarial_loss * generator_loss(fake_output)

      total_generator_loss = (
        gan_reconstruction_loss + icr_consistency_loss  + local_generator_loss + global_generator_loss +
        f_rec_loss
      )

    gradients_of_generator = generator_tape.gradient(total_generator_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    gradients_of_global_discriminator = global_disc_tape.gradient(global_discriminator_loss, discriminator.trainable_variables)
    global_discriminator_optimizer.apply_gradients(zip(gradients_of_global_discriminator, discriminator.trainable_variables))

    gradients_of_local_discriminator = local_discriminator_tape.gradient(local_discriminator_loss, local_discriminator.trainable_variables)
    local_discriminator_optimizer.apply_gradients(zip(gradients_of_local_discriminator, local_discriminator.trainable_variables))

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
        "total/global_disc_loss": global_discriminator_loss,
        "total/local_disc_loss": local_discriminator_loss,
        "generator/reconstruction_loss": (gan_reconstruction_loss),
        "generator/landmarks_consistency_loss": (icr_landmark_loss),
        "generator/mask_consistency_loss": (icr_face_mask_loss),
        "generator/local_loss": (local_generator_loss),
        "generator/global_loss": (global_generator_loss),
      }
    }

def train(dataset, epochs):
  total_steps = 0
  for epoch in range(epochs):
    start = time.time()
    batch_of_masks = mask_rgb(batch_size)
    for step, image_batch in enumerate(dataset):
      values = train_step(image_batch, batch_of_masks)
      total_steps += 1
      if total_steps % config["train"]["log_interval"] == 0:
        print("epoch %d step %d" % (epoch + 1, step + 1))

      if total_steps % config["train"]["log_interval"] == 0:
        with writer.as_default():
          for name, value in values["losses"].items():
            tf.summary.scalar(name, value, step=total_steps)
          # TODO: Improve image saving
          # original_images = (values["outputs"]["original_images"] + 1.) / 2.
          # reconstructed_images = (values["outputs"]["reconstructed_images"] + 1.) / 2.
          # landmark_reconstructed = tf.sigmoid(values["outputs"]["landmark_reconstructed"])
          # face_mask_reconstructed = tf.sigmoid(values["outputs"]["face_mask_reconstructed"])
          # face_part_reconstructed = (values["outputs"]["face_part_reconstructed"] + 1.) / 2.
          # tf.summary.image("original_images", original_images, step=total_steps)
          # tf.summary.image("reconstructed_images", reconstructed_images, step=total_steps)
          # tf.summary.image("landmark_reconstructed", landmark_reconstructed, step=total_steps)
          # tf.summary.image("face_mask_reconstructed", face_mask_reconstructed, step=total_steps)
          # tf.summary.image("face_part_reconstructed", face_part_reconstructed, step=total_steps)
            

      if total_steps % config["train"]["save_interval"] == 0:
        mask_batch = mask_rgb(num_examples_to_generate)
        z_seed = tf.random.normal([num_examples_to_generate, noise_dim])
        for item in train_dataset.take(1):
          sample = item[:,:,:,0:3] * (1. - mask_batch)
        tf.print("losses", values["losses"])	
        predictions = inference(pfcGan.landmark_encoder, 
                                pfcGan.landmark_decoder, 
                                pfcGan.face_encoder, 
                                pfcGan.face_mask_decoder, 
                                pfcGan.face_part_decoder, 
                                pfcGan.generator, 
                                z = z_seed,
                                batch_incomplete = sample,
                                batch_mask = mask_batch[:,:,:,0:1])
        # predictions = (predictions[i] * args[2][i]) + (args[1][i] * (1. - args[2][i]))
        checkpoint.save(file_prefix = checkpoint_prefix)
        generate_and_save_images(predictions = predictions,
                                 original=sample,
                                 mask=mask_batch,
                                 step = total_steps + 1)
        # Slep 4 minutes
        # time.sleep(240)
        if config["utils"]["show_embedding"]:
          display.clear_output(wait=True)
          outputs = values["outputs"]
          original_image = outputs["original_images"][0]
          landmark_sample = outputs["landmark_reconstructed"][0]
          mask_sample = outputs["face_mask_reconstructed"][0]
          face_part_sample = outputs["face_part_reconstructed"][0]
          reconstructed_image = outputs["reconstructed_images"][0]
          mask = batch_of_masks[0]
          fig = plt.figure(figsize=(8, 6))

          # Subplot 1: Landmark sample
          plt.subplot(2, 3, 1)
          plt.imshow(tf.sigmoid(landmark_sample), cmap='gray')
          plt.title("Landmark")
          plt.axis('off')

          # Subplot 2: Mask sample
          plt.subplot(2, 3, 2)
          plt.imshow(tf.sigmoid(mask_sample), cmap='gray')
          plt.title("Mask")
          plt.axis('off')

          # Subplot 3: Face part sample
          plt.subplot(2, 3, 3)
          plt.imshow((face_part_sample + 1.) / 2.)
          plt.title("Face Part")
          plt.axis('off')

          # Subplot 4: Original image
          plt.subplot(2, 3, 4)
          plt.imshow((original_image + 1.) / 2.)
          plt.title("Original")
          plt.axis('off')

          # Subplot 5: Reconstructed image
          plt.subplot(2, 3, 5)
          plt.imshow((reconstructed_image + 1.) / 2.)
          plt.title("Reconstructed")
          plt.axis('off')

          # Subplot 6: Mask
          plt.subplot(2, 3, 6)
          plt.imshow(mask, cmap='gray')
          plt.title("Random Mask")
          plt.axis('off')

          plt.show()
          plt.close()



    # Save the model every 15 epochs
    # if (epoch + 1) % 15 == 0:

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # # Generate after the final epoch
  # display.clear_output(wait=True)
  # generate_and_save_images(generator,
  #                          epoch = epochs,
  #                          args =seed)


## Train the model
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(train_dataset, EPOCHS)
