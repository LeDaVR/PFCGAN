### Setup
import tensorflow as tf

import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

from IPython import display

from data import MultiChannelDataLoader
from model import make_extractor_model, make_generator_model, make_discriminator_model, \
    make_landmark_encoder, make_landmark_decoder, \
    make_face_encoder, make_face_mask_decoder, make_face_part_decoder, make_local_discriminator, CyclicalAnnealingScheduler
from utils import generate_and_save_images, mask_rgb

import yaml

# Cargar configuración desde el archivo YAML
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Acceso a las rutas y configuraciones
original_img_dir = config["paths"]["original_img_dir"]
feature_img_dir = config["paths"]["feature_img_dir"]
### Load and prepare the dataset

# Hyperparameters
EPOCHS = config["hyper_parameters"]["epochs"]
batch_size = config["hyper_parameters"]["batch_size"]
w_landmarks = 2000
w_face_mask = 4000
w_face_part = 6000
# f_kl = 0.2
# kl_embedding = 0.5
# e_kl = 1.
consistency_loss = 100.
landmark_factor = 1.
mask_factor = 1.
annealing_steps = 2000
max_beta = .5

# train_dataset = create_image_dataset(original_img_dir, feature_img_dir, batch_size=batch_size)
data_loader = MultiChannelDataLoader(original_img_dir, feature_img_dir, img_size=(128, 128))
train_dataset = data_loader.create_dataset(batch_size=batch_size)
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

scheduler = CyclicalAnnealingScheduler(cycle_length=annealing_steps, max_beta=max_beta, min_beta=0.0, n_cycles=4)

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
  l1 = tf.reduce_mean(tf.abs(y_true - x), axis = [1,2,3])
  return tf.reduce_mean(l1)

def reconstruction_loss_with_logits(x_logit, y_true, weight=tf.constant(7.)):
    cross_ent = tf.nn.weighted_cross_entropy_with_logits(
        logits=x_logit, 
        labels=y_true, 
        pos_weight=weight
    )
    imgs_loss = tf.reduce_mean(cross_ent, axis=[1, 2, 3])
    return tf.reduce_mean(imgs_loss)

def kl_divergence_loss(mean, logvar):
    kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
    kl_loss = tf.reduce_mean(kl_loss)
    return kl_loss

def masked_loss(y_true, y_pred, mask):
   l1 = (tf.abs(y_true - y_pred))
   masked_l1 = l1 * mask
   total_img_error = tf.reduce_sum(masked_l1, axis=[1,2,3])
   num_pixels_per_image = tf.reduce_sum(mask, axis=[1, 2, 3])  
   normalized_img_error = total_img_error / (num_pixels_per_image + 1e-8)
   return tf.reduce_mean(normalized_img_error)

def masked_weighted_cross_entropy_loss(x_logits, y_true, mask, weight=tf.constant(7.)):
    cross_entropy_loss = tf.nn.weighted_cross_entropy_with_logits(logits=x_logits, labels=y_true, pos_weight=weight)
    masked_cross_entropy_loss = cross_entropy_loss * mask
    total_img_error = tf.reduce_sum(masked_cross_entropy_loss, axis=[1,2,3])
    num_pixels_per_image = tf.reduce_sum(mask, axis=[1, 2, 3])
    normalized_img_error = total_img_error / (num_pixels_per_image + 1e-8)
    return tf.reduce_mean(normalized_img_error)

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

# def inference(lencoder, ldecoder, fencoder, mdecoder, fdecoder, generator, z, batch_incomplete, batch_mask):
#     l_mu, l_log_var = lencoder([batch_incomplete, z, batch_mask], training=False)
#     reparametrized_landmarks = reparametrize(l_mu, l_log_var)
#     f_mu, f_log_var = fencoder([batch_incomplete, z, batch_mask], training=False)
#     reparametrized_face= reparametrize(f_mu, f_log_var)
#     emb = tf.concat([reparametrized_landmarks, reparametrized_face], axis=-1)
#     fake = generator([emb, batch_incomplete, batch_mask], training=False)
#     return fake

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
def train_step(batch, lbatch_mask, kl_val):
    with  tf.GradientTape() as embedding_tape:
      # Prepare the batch
      batch_size = tf.shape(batch)[0]
      # lbatch_mask = mask_rgb(batch_size)
      tbatch_original = batch[:,:,:,0:3]
      tbatch_original_incomplete = tbatch_original * (1. - lbatch_mask)
      tbatch_landmarks = batch[:,:,:,3:4]
      tbatch_face_mask = batch[:,:,:,4:5]
      tbatch_face_part = batch[:,:,:,5:8]

      one_channel_mask = lbatch_mask[:,:,:,0:1]

      prob_landmarks = ( tbatch_landmarks + 1. ) / 2.
      prob_face_mask = ( tbatch_face_mask + 1. ) / 2.

      one_channel_inverted_mask = 1. - one_channel_mask
      rgb_inverted_mask = 1. - lbatch_mask

      # Extractor 

      z_random = tf.random.normal(shape = [batch_size, noise_dim])

      _, zf_emb, zf_landmarks, zf_mask, zf_part = feature_embedding(tbatch_original_incomplete, z_random, one_channel_mask, training=True)
      # zer_mu, zer_log_var = extractor(tf.concat([zf_landmarks, zf_mask, zf_part], axis=-1), training=True)
      # zer_sample = reparametrize(zer_mu, zer_log_var)
      # _, _ , zer_landmarks, zer_mask, zer_part = feature_embedding(tbatch_original_incomplete, zer_sample, one_channel_mask, training=True)

      # Calculate reconstruction loss for the unmasked areas
      zf_landmarks_im = tf.sigmoid(zf_landmarks) 
      zf_mask_im = tf.sigmoid(zf_mask)
      # zer_landmarks_im = tf.sigmoid(zer_landmarks) * 2. - 1.
      # zer_mask_im = tf.sigmoid(zer_mask) * 2. - 1.

      zf_loss = (
        w_landmarks * masked_loss(zf_landmarks_im, prob_landmarks, one_channel_inverted_mask) +
        w_face_mask * masked_loss(zf_mask_im, prob_face_mask, one_channel_inverted_mask) +
        w_face_part * masked_loss(zf_part, tbatch_face_part, rgb_inverted_mask)
      )

      # Extractor loss
      extractor_consistency_loss = zf_loss

      # Face Embedding
      e_mu, e_log_var = extractor(tf.concat([tbatch_landmarks, tbatch_face_mask, tbatch_face_part], axis=-1), training=True)
      extractor_sample = reparametrize(e_mu, e_log_var)
      extractor_kl_loss = kl_val *  kl_divergence_loss(e_mu, e_log_var)

      total_extractor_loss = extractor_kl_loss + extractor_consistency_loss

      # (z1,z2), zr_emb, zr_landmarks, zr_mask, zr_part = feature_embedding(batch_original_incomplete, extractor_sample, mask_batch[:,:,:,0:1])

      # Landkard encoder
      zlr_mu, zlr_log_var  = landmark_encoder([tbatch_original_incomplete, extractor_sample,  one_channel_mask], training=True)
      zr_l_sample = reparametrize(zlr_mu, zlr_log_var)
      icr_landmarks = landmark_decoder(zr_l_sample, training=True)
      # Mask encoder
      zfr_mu, zfr_log_var  = face_encoder([tbatch_original_incomplete,extractor_sample, one_channel_mask], training=True)
      zfr_f_sample = reparametrize(zfr_mu, zfr_log_var)
      z_emb = tf.concat([zr_l_sample, zfr_f_sample], axis=-1)
      # Face mask decoder
      icr_face_mask = face_mask_decoder(z_emb, training=True)
      icr_face_part = face_part_decoder(z_emb, training=True)

      landmark_reconstruction_loss = w_landmarks *  reconstruction_loss_with_logits(icr_landmarks, prob_landmarks, (landmark_factor))
      face_mask_reconstruction_loss = w_face_mask * reconstruction_loss_with_logits(icr_face_mask, prob_face_mask, (mask_factor))
      face_part_reconstruction_loss = w_face_part * l1_reconstruction_loss(icr_face_part, tbatch_face_part)
      embedding_reconstruction_loss = landmark_reconstruction_loss +  face_mask_reconstruction_loss + face_part_reconstruction_loss
      
      # Embedding kl loss
      z1_kl_loss = kl_val * kl_divergence_loss(zlr_mu, zlr_log_var)
      z2_kl_loss = kl_val *  kl_divergence_loss(zfr_mu, zfr_log_var)

      total_embedding_loss = embedding_reconstruction_loss + z1_kl_loss + z2_kl_loss + total_extractor_loss 

    face_embedding_trainable_variables = (
      extractor.trainable_variables + 
      face_encoder.trainable_variables + 
      landmark_encoder.trainable_variables +
      face_mask_decoder.trainable_variables + 
      face_part_decoder.trainable_variables+ 
      landmark_decoder.trainable_variables  
    )

    gradients_of_embedding = embedding_tape.gradient(total_embedding_loss, face_embedding_trainable_variables)
    clipped_gradients = [tf.clip_by_norm(g, 1.0) for g in gradients_of_embedding]
    face_embedding_optimizer.apply_gradients(zip(clipped_gradients, face_embedding_trainable_variables))

    return {
      "outputs": {
        "original_images": tbatch_original,
        "landmark_original": tbatch_landmarks,
        "face_mask_original": tbatch_face_mask,
        "face_part_original": tbatch_face_part,
        "landmark_reconstructed": icr_landmarks,
        "face_mask_reconstructed": icr_face_mask,
        "face_part_reconstructed": icr_face_part,
      },
      "losses": {	
        "total/total_embedding_loss": total_embedding_loss,
        "total/total_extractor_loss": total_extractor_loss,
        "extractor/kl_loss": (extractor_kl_loss),
        "extractor/consistency_loss": extractor_consistency_loss,
        "embedding/landmark_reconstruction_loss": landmark_reconstruction_loss,
        "embedding/face_mask_reconstruction_loss": face_mask_reconstruction_loss,
        "embedding/face_part_reconstruction_loss": face_part_reconstruction_loss,
        "embedding/fake_reconstruction_loss": zf_loss,
        "embedding/z1_kl_loss": (z1_kl_loss),
        "embedding/z2_kl_loss": (z2_kl_loss),
      }
    }

def train(dataset, epochs):
  total_steps = 0
  for epoch in range(epochs):
    start = time.time()
    batch_of_masks = mask_rgb(batch_size)
    for step, image_batch in enumerate(dataset):
      # Generate a batch of masks every 100 steps
      if step % 100 == 0:
        batch_of_masks = mask_rgb(batch_size)
      kl_val = scheduler.get_beta(total_steps)
      values = train_step(image_batch, batch_of_masks, kl_val)
      total_steps += 1
      if total_steps % config["train"]["log_interval"] == 0:
        print("epoch %d step %d" % (epoch + 1, step + 1))

      if total_steps % config["train"]["log_interval"] == 0:
        with writer.as_default():
          for name, value in values["losses"].items():
            tf.summary.scalar(name, value, step=total_steps)

      if total_steps % config["train"]["save_interval"] == 0:
        # mask_batch = mask_rgb(num_examples_to_generate)
        # z_seed = tf.random.normal([num_examples_to_generate, noise_dim])
        # for item in train_dataset.take(1):
        #   sample = item[:,:,:,0:3] * (1. - mask_batch)
        tf.print("losses", values["losses"])	
        # predictions = inference(pfcGan.landmark_encoder, 
        #                         pfcGan.landmark_decoder, 
        #                         pfcGan.face_encoder, 
        #                         pfcGan.face_mask_decoder, 
        #                         pfcGan.face_part_decoder, 
        #                         pfcGan.generator, 
        #                         z = z_seed,
        #                         batch_incomplete = sample,
        #                         batch_mask = mask_batch[:,:,:,0:1])
        # predictions = (predictions[i] * args[2][i]) + (args[1][i] * (1. - args[2][i]))
        checkpoint.save(file_prefix = checkpoint_prefix)
        # generate_and_save_images(predictions = predictions,
        #                          original=sample,
        #                          mask=mask_batch,
        #                          step = total_steps + 1)
        # Slep 4 minutes
        # time.sleep(240)
        if config["utils"]["show_embedding"]:
          display.clear_output(wait=True)
          outputs = values["outputs"]
          original_image = outputs["original_images"][0]
          landmark_sample = outputs["landmark_reconstructed"][0]
          mask_sample = outputs["face_mask_reconstructed"][0]
          face_part_sample = outputs["face_part_reconstructed"][0]

          landmark_original = outputs["landmark_original"][0]
          face_mask_original = outputs["face_mask_original"][0]
          face_part_original = outputs["face_part_original"][0]

          # shift from -1 to 1
          original_image = (original_image + 1.) / 2.
          landmark_sample = tf.sigmoid(landmark_sample)
          mask_sample = tf.sigmoid(mask_sample)
          face_part_sample = (face_part_sample + 1.) / 2.

          # shift original from -1 to 1
          landmark_original = (landmark_original + 1.) / 2.
          face_mask_original = (face_mask_original + 1.) / 2.
          face_part_original = (face_part_original + 1.) / 2.

          #Mask original
          original_image *= (1. - batch_of_masks[0])

          # Plot vs for landmarks, masks and parts

          # Crear el gráfico con varias subgráficas
          fig, axes = plt.subplots(3, 3, figsize=(12, 12))

          # Títulos para cada subgráfico
          titles = [
              "Original Image", "Landmark Sample", "Mask Sample",
              "Face Part Sample", "Landmark Original", "Face Mask Original",
              "Face Part Original", "Landmark vs Sample", "Sample vs Mask"
          ]

          # Muestra las imágenes
          axes[0, 0].imshow(original_image)
          axes[0, 0].set_title(titles[0])
          axes[0, 1].imshow(landmark_sample, cmap='gray')
          axes[0, 1].set_title(titles[1])
          axes[0, 2].imshow(mask_sample, cmap='gray')
          axes[0, 2].set_title(titles[2])

          axes[1, 0].imshow(face_part_sample)
          axes[1, 0].set_title(titles[3])
          axes[1, 1].imshow(landmark_original, cmap='gray')
          axes[1, 1].set_title(titles[4])
          axes[1, 2].imshow(face_mask_original, cmap='gray')
          axes[1, 2].set_title(titles[5])

          axes[2, 0].imshow(face_part_original)
          axes[2, 0].set_title(titles[6])

          # Genera comparaciones entre las imágenes
          axes[2, 1].imshow(landmark_sample - landmark_original, cmap='hot')
          axes[2, 1].set_title(titles[7])

          axes[2, 2].imshow(mask_sample - face_mask_original, cmap='hot')
          axes[2, 2].set_title(titles[8])

          # Ajusta el espacio entre las subgráficas
          plt.tight_layout()
          plt.savefig('res/embedding_at_step_{:04d}.png'.format(total_steps))
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
