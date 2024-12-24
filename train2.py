### Setup
import tensorflow as tf
import glob

import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

from data import MultiChannelDataLoader, load_image_dataset
from model import make_extractor_model, make_generator_model, make_discriminator_model, \
    make_landmark_encoder, make_landmark_decoder, \
    make_face_encoder, make_face_mask_decoder, make_face_part_decoder, make_local_discriminator,\
    WGAN_GP
from utils import generate_and_save_images, mask_rgb
from latent_clasiffier import LatentClassifier, latent_discriminator_loss, latent_generator_loss

import yaml

# tf.config.run_functions_eagerly(True)
# tf.debugging.enable_check_numerics()

# Cargar configuración desde el archivo YAML
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Acceso a las rutas y configuraciones
original_img_dir = config["paths"]["original_img_dir"]
feature_img_dir = config["paths"]["feature_img_dir"]
out_train_interval = config["train"]["out_save_interval"]
### Load and prepare the dataset

# Hyperparameters
EPOCHS = config["hyper_parameters"]["epochs"]
batch_size = config["hyper_parameters"]["batch_size"]
w_landmarks = 2000.
w_face_mask = 4000.
w_face_part = 4000.
adversarial_loss = 20.
latent_classifier_beta = 1.
rec_loss = 40.
# f_kl = 0.2
# kl_embedding = 0.5
# e_kl = 1.
consistency_loss = 100.
landmark_factor = 1.
mask_factor = 1.

original_files = sorted(glob.glob(os.path.join(original_img_dir, "*.jpg")))
#Instead of loading from directory just parse original files onto feature files
landmarks_files = sorted(glob.glob(os.path.join(feature_img_dir, "*_landmarks.jpg")))
face_mask_files = sorted(glob.glob(os.path.join(feature_img_dir, "*_mask.jpg")))
face_part_files = sorted(glob.glob(os.path.join(feature_img_dir, "*_face_part.jpg")))

# train_dataset = create_image_dataset(original_img_dir, feature_img_dir, batch_size=batch_size)
# data_loader = MultiChannelDataLoader(original_img_dir, feature_img_dir, img_size=(128, 128))
# train_dataset = data_loader.create_dataset(batch_size=batch_size)

original_dataset = load_image_dataset(original_files, num_channels=3, binarize=False, batch_size=batch_size)
landmarks_dataset = load_image_dataset(landmarks_files, num_channels=1, binarize=True, threshold=0, batch_size=batch_size)
face_mask_dataset = load_image_dataset(face_mask_files, num_channels=1, binarize=True, threshold=0, batch_size=batch_size)
face_part_dataset = load_image_dataset(face_part_files, num_channels=3, binarize=False, batch_size=batch_size)

train_dataset = tf.data.Dataset.zip((original_dataset, landmarks_dataset, face_mask_dataset, face_part_dataset))

class PFCGAN():
   def __init__(self, landmark_encoder, landmark_decoder, face_encoder, face_mask_decoder, face_part_decoder, generator, latent_classifier):
      self.landmark_encoder = landmark_encoder
      self.landmark_decoder = landmark_decoder
      self.face_encoder = face_encoder
      self.face_mask_decoder = face_mask_decoder
      self.face_part_decoder = face_part_decoder
      self.generator = generator
      self.latent_classifier = latent_classifier


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

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

def ce_loss_with_logits(x_logit, y_true, weight=tf.constant(7.)):
    cross_ent = tf.nn.weighted_cross_entropy_with_logits(
        logits=x_logit, 
        labels=y_true, 
        pos_weight=weight
    )
    total_white_pixels = tf.reduce_sum(y_true, axis=[1, 2, 3])
    total_black_pixels = tf.reduce_sum(1. - y_true, axis=[1, 2, 3])
    white_loss = tf.reduce_sum(cross_ent * y_true, axis=[1, 2, 3]) / total_white_pixels
    black_loss = tf.reduce_sum(cross_ent * (1. - y_true), axis=[1, 2, 3]) / total_black_pixels

    total_loss = ( white_loss + black_loss ) / 2.
    return tf.reduce_mean(total_loss)

# def kl_divergence_loss(mean, logvar):
#     kl_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
#     kl_loss = tf.reduce_mean(kl_loss)
#     return kl_loss

def masked_loss(y_true, y_pred, mask):
   l1 = (tf.abs(y_true - y_pred))
   masked_l1 = l1 * mask
   total_img_error = tf.reduce_sum(masked_l1, axis=[1,2,3])
   num_pixels_per_image = tf.reduce_sum(mask, axis=[1, 2, 3])  
   normalized_img_error = total_img_error / (num_pixels_per_image + 1e-8)
   return tf.reduce_mean(normalized_img_error)

def masked_ce_loss_with_logits(x_logits, y_true, mask, weight=tf.constant(7.)):
    cross_entropy_loss = tf.nn.weighted_cross_entropy_with_logits(logits=x_logits, labels=y_true, pos_weight=weight)
    masked_cross_entropy_loss = cross_entropy_loss * mask
    masked_ytrue = y_true * mask
    total_white_pixels = tf.reduce_sum(masked_ytrue, axis=[1, 2, 3])
    total_black_pixels = tf.reduce_sum(1. - masked_ytrue, axis=[1, 2, 3])
    white_loss = tf.reduce_sum(masked_cross_entropy_loss * masked_ytrue, axis=[1, 2, 3]) / total_white_pixels
    black_loss = tf.reduce_sum(masked_cross_entropy_loss * (1. - masked_ytrue), axis=[1, 2, 3]) / total_black_pixels

    total_loss = ( white_loss + black_loss ) / 2.
    return tf.reduce_mean(total_loss)
    total_img_error = tf.reduce_sum(masked_cross_entropy_loss, axis=[1,2,3])
    num_pixels_per_image = tf.reduce_sum(mask, axis=[1, 2, 3])
    normalized_img_error = total_img_error / (num_pixels_per_image + 1e-8)
    return tf.reduce_mean(normalized_img_error)

extractor = make_extractor_model()
landmark_encoder = make_landmark_encoder()
landmark_decoder = make_landmark_decoder()
face_encoder = make_face_encoder()
face_mask_decoder = make_face_mask_decoder()
face_part_decoder = make_face_part_decoder()
latent_classifier = LatentClassifier(latent_dim=256)
latent_classifier512 = LatentClassifier(latent_dim=512)
latent_classifier_face = LatentClassifier(latent_dim=256)
wpgan = WGAN_GP()
pfcGan = PFCGAN(landmark_encoder, landmark_decoder, face_encoder, face_mask_decoder, face_part_decoder, wpgan.generator, latent_classifier)

face_embedding_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
latent_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
latent_face_discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
latent_discriminator_optimizer512 = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

### Save checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
                                
                                # generator_optimizer=generator_optimizer,
                                #  extractor_optimizer=extractor_optimizer,
                                global_discriminator_optimizer=wpgan.dglobal_optimizer,
                                local_discriminator_optimizer=wpgan.dlocal_optimizer,
                                generator_optimizer=wpgan.g_optimizer,
                                generator=wpgan.generator,
                                discriminator=wpgan.discriminator,
                                local_discriminator=wpgan.local_discriminator,
                                face_embedding_optimizer=face_embedding_optimizer,
                                latent_discriminator_optimizer=latent_discriminator_optimizer,
                                latent_face_discriminator_optimizer=latent_face_discriminator_optimizer,
                                latent_discriminator_optimizer512=latent_discriminator_optimizer512,
                                latent_classifier=latent_classifier,
                                latent_classifier512=latent_classifier512,
                                extractor=extractor,
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

  return (zl_mu, zl_log_var), (zf_mu, zf_log_var),  (landmark_sample, face_sample), z_emb, landmark_reconstructed, face_mask_reconstructed, face_part_reconstructed

noise_dim = 512
num_examples_to_generate = batch_size

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(log_dir)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(batch, lbatch_mask):
    with  tf.GradientTape() as embedding_tape, \
      tf.GradientTape() as latent_discriminator_tape:
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

      _, _, _, zf_emb, zf_landmarks, zf_mask, zf_part = feature_embedding(tbatch_original_incomplete, z_random, one_channel_mask, training=True)
      # zf_reconstructed = generator([zf_emb, tbatch_original_incomplete, one_channel_mask], training=True)

      # zf_reconstructed_loss = rec_loss * masked_loss(zf_reconstructed, tbatch_original, rgb_inverted_mask)
      zf_landmarks_loss = w_landmarks * masked_ce_loss_with_logits(zf_landmarks, prob_landmarks, one_channel_inverted_mask, weight=landmark_factor)
      zf_mask_loss = w_face_mask * masked_ce_loss_with_logits(zf_mask, prob_face_mask, one_channel_inverted_mask, weight=mask_factor)
      zf_part_loss = w_face_part * masked_loss(zf_part, tbatch_face_part, rgb_inverted_mask)

      zf_loss = zf_landmarks_loss + zf_mask_loss + zf_part_loss

      #Skip the regression loss for the momment

      zer_mu, zer_log_var = extractor([batch[:,:,:,3:8]], training=True)
      zer_sample = reparametrize(zer_mu, zer_log_var)

      (zerl_mu, zerl_log_var), (zerf_mu, zerf_log_var), (zerl_sample, zerf_sample) , zer_emb, zer_landmarks, zer_mask, zer_part = feature_embedding(tbatch_original_incomplete, zer_sample, one_channel_mask, training=True)
      # zer_reconstructed = generator([zer_emb, tbatch_original_incomplete, one_channel_mask], training=True)
      # zer_reconstructed_loss = rec_loss * l1_reconstruction_loss(zer_reconstructed, tbatch_original)
      zer_landmarks_loss = w_landmarks * ce_loss_with_logits(zer_landmarks, prob_landmarks, weight=landmark_factor)
      zer_mask_loss = w_face_mask * ce_loss_with_logits(zer_mask, prob_face_mask, weight=mask_factor)
      zer_part_loss = w_face_part * l1_reconstruction_loss(zer_part, tbatch_face_part)

      zer_loss = zer_landmarks_loss + zer_mask_loss + zer_part_loss

      # extractor_kl_loss =kl_divergence_loss(zer_mu, zer_log_var)
      samples512 = tf.random.normal(shape=[batch_size, 512])
      samples = tf.random.normal(shape=[batch_size, 256])

      clas512_out_real = latent_classifier512(samples512)
      zer_out = latent_classifier512(zer_sample)
      emb_out = latent_classifier512(zer_emb)
      class256_out_real = latent_classifier(samples)
      zerl_out = latent_classifier(zerl_sample)
      zerf_out = latent_classifier_face(zerf_sample)

      # Generate normal dsitribution samples
      # use latent discriminator instead of kl loss
      e_lc_loss = latent_discriminator_loss(clas512_out_real , zer_out)
      l_l_loss = latent_discriminator_loss(class256_out_real , zerl_out)
      l_f_loss = latent_discriminator_loss(class256_out_real , zerf_out)
      emb_lc_loss = latent_discriminator_loss(clas512_out_real , emb_out)

      classifier_loss = (l_l_loss + l_f_loss + emb_lc_loss + e_lc_loss)
      
      classifier_e_loss = latent_classifier_beta *  latent_generator_loss(zer_out )
      classifier_l_loss = latent_classifier_beta *  latent_generator_loss(zerl_out)
      classifier_f_loss = latent_classifier_beta *  latent_generator_loss(zerf_out )
      classifier_zemb_loss = latent_classifier_beta *  latent_generator_loss(emb_out )

      total_classifier_gen_loss = classifier_e_loss + classifier_l_loss + classifier_f_loss + classifier_zemb_loss

      # real_output =  discriminator(tbatch_original, training=True)  
      # fake_output =  discriminator(zf_reconstructed, training=True)

      # Local discriminator loss
      masked_batch_original = tbatch_original * (lbatch_mask)
      # masked_generated_images = zf_reconstructed * (lbatch_mask)

      # local_real_output = local_discriminator([masked_batch_original, one_channel_mask], training=True)  
      # local_fake_output = local_discriminator([masked_generated_images, one_channel_mask], training=True)

      # global_discriminator_loss = discriminator_loss(real_output, fake_output)
      # local_discriminator_loss = discriminator_loss(local_real_output, local_fake_output)
      #
      # local_generator_loss = adversarial_loss * generator_loss(local_fake_output)
      # global_generator_loss = adversarial_loss * generator_loss(fake_output)
      
      total_embedding_loss = zf_loss + zer_loss + total_classifier_gen_loss

      # total_rec_loss = zer_reconstructed_loss + zf_reconstructed_loss
      # total_generator_loss = local_generator_loss + global_generator_loss + total_rec_loss

    face_embedding_trainable_variables = (
      extractor.trainable_variables + 
      face_encoder.trainable_variables + 
      landmark_encoder.trainable_variables +
      face_mask_decoder.trainable_variables + 
      face_part_decoder.trainable_variables+ 
      landmark_decoder.trainable_variables
    )

    classfifiers_trainable_variables = (
      latent_classifier.trainable_variables + 
      latent_classifier512.trainable_variables +
      latent_classifier_face.trainable_variables
    )

    gradients_of_embedding = embedding_tape.gradient(total_embedding_loss, face_embedding_trainable_variables)
    # clipped_gradients = [tf.clip_by_norm(g, 1.0) for g in gradients_of_embedding]
    face_embedding_optimizer.apply_gradients(zip(gradients_of_embedding, face_embedding_trainable_variables))

    # gradients_of_global_discriminator = global_disc_tape.gradient(global_discriminator_loss, discriminator.trainable_variables)
    # global_discriminator_optimizer.apply_gradients(zip(gradients_of_global_discriminator, discriminator.trainable_variables))
    #
    # gradients_of_local_discriminator = local_discriminator_tape.gradient(local_discriminator_loss, local_discriminator.trainable_variables)
    # local_discriminator_optimizer.apply_gradients(zip(gradients_of_local_discriminator, local_discriminator.trainable_variables))

    gradients_of_latent_discriminator = latent_discriminator_tape.gradient(classifier_loss, classfifiers_trainable_variables)
    latent_discriminator_optimizer.apply_gradients(zip(gradients_of_latent_discriminator, classfifiers_trainable_variables))

    # gradients_of_generator = generator_tape.gradient(total_generator_loss, generator.trainable_variables)
    # generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return {
      "outputs": {
        "original_images": tbatch_original,
        "landmark_original": tbatch_landmarks,
        "face_mask_original": tbatch_face_mask,
        "face_part_original": tbatch_face_part,
        # "reconstructed_images": zer_reconstructed,
        "landmark_reconstructed": zer_landmarks,
        "face_mask_reconstructed": zer_mask,
        "face_part_reconstructed": zer_part,
      },
      "losses": {	
        "total/total_embedding_loss": total_embedding_loss,
        # "total/total_generator_loss": total_generator_loss,

        # "extractor/consistency_loss": extractor_consistency_loss,
        "embedding/zer_landmarks_loss": zer_landmarks_loss,
        "embedding/zer_mask_loss": zer_mask_loss,
        "embedding/zer_part_loss": zer_part_loss,
        "embedding/zf_landmarks_loss": zf_landmarks_loss,
        "embedding/zf_mask_loss": zf_mask_loss,
        "embedding/zf_part_loss": zf_part_loss,
        "embedding/classifier_loss": total_classifier_gen_loss,
        "embedding/e_lc_loss": classifier_e_loss,
        "embedding/l_l_loss": classifier_l_loss,
        "embedding/f_l_loss": classifier_f_loss,
        "embedding/emb_loss": classifier_zemb_loss,
        

        # "generator/global_loss": global_generator_loss,
        # "generator/local_loss": local_generator_loss,
        # "generator/generator_loss": total_generator_loss,
        # "generator/zer_reconstruction_loss": zer_reconstructed_loss,
        # "generator/zf_reconstruction_loss": zf_reconstructed_loss,

        # "discriminator/global_discriminator_loss": global_discriminator_loss,
        # "discriminator/local_discriminator_loss": local_discriminator_loss,
        "discriminator/l_kl_loss": l_l_loss,
        "discriminator/f_kl_loss": l_f_loss,
        "discriminator/emb_kl_loss": emb_lc_loss,
        "discriminator/e_lc_loss": e_lc_loss,
      }
    }

def get_zemb(image_batch, batch_of_masks):
  original_images = image_batch[:,:,:,0:3]
  original_features = image_batch[:,:,:,3:8]
  batch_incomplete = original_images * (1. - batch_of_masks)
  one_channel_mask = batch_of_masks[:,:,:,0:1]
  zer_mu, zer_log_var = extractor([ original_features ], training=False)
  zer_sample = reparametrize(zer_mu, zer_log_var)
  _, _, _ , zer_emb, _, _, _ = feature_embedding(batch_incomplete, zer_sample, one_channel_mask, training=False)
  return zer_emb

def train(dataset, epochs):
  total_steps = 0
  for epoch in range(epochs):
    start = time.time()
    batch_of_masks = mask_rgb(batch_size)
    for step, image_batch in enumerate(dataset):
      # Generate a batch of masks every 100 steps
      if step % 100 == 0:
        batch_of_masks = mask_rgb(batch_size)

      original, landmarks, face_mask, face_part = image_batch
      one_tensor_batch = tf.concat([original, landmarks, face_mask, face_part], axis=-1)

      values = train_step(one_tensor_batch, batch_of_masks)

      z_emb = get_zemb(one_tensor_batch, batch_of_masks)
      imcomplete = original * (1. - batch_of_masks)
      # Execute feature embedding for getting inputs for the generator
      # tf.print(tf.shape(original))
      # tf.print(tf.shape(imcomplete))
      # tf.print(tf.shape(batch_of_masks))
      # gdloss, ldloss, genloss ,generated_images = wpgan.train_step(original, imcomplete, batch_of_masks, z_emb)

      total_steps += 1
      if total_steps % config["train"]["log_interval"] == 0:
        tf.print("epoch %d step %d" % (epoch + 1, step + 1))

      if total_steps % config["train"]["log_interval"] == 0:
        with writer.as_default():
          for name, value in values["losses"].items():
            tf.summary.scalar(name, value, step=total_steps)
          # tf.summary.scalar("generator/global_loss", gdloss, step=total_steps)
          # tf.summary.scalar("generator/local_loss", ldloss, step=total_steps)
          # tf.summary.scalar("generator/gen_loss", genloss, step=total_steps)

      if total_steps % config["train"]["save_interval"] == 0:
        tf.print("losses", values["losses"])	
        checkpoint.save(file_prefix = checkpoint_prefix)
      if total_steps % out_train_interval == 0 and config["utils"]["show_embedding"]:
          outputs = values["outputs"]
          original_image = outputs["original_images"][0]
          reconstructed_image = tf.zeros_like(original_image)
          # reconstructed_image = generated_images[0]
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
          reconstructed_image = (reconstructed_image + 1.) / 2.

          # shift original from -1 to 1
          landmark_original = (landmark_original + 1.) / 2.
          face_mask_original = (face_mask_original + 1.) / 2.
          face_part_original = (face_part_original + 1.) / 2.

          #Mask original
          original_image *= (1. - batch_of_masks[0])

          # Plot vs for landmarks, masks and parts

          # Crear el gráfico con varias subgráficas
          fig, axes = plt.subplots(2, 5, figsize=(12, 12))

          # Títulos para cada subgráfico
          titles = [
              "Original Image", "Reconstructed Image", "Landmark Sample", "Mask Sample",
              "Face Part Sample", "Landmark Original", "Face Mask Original",
              "Face Part Original", "Landmark vs Sample", "Sample vs Mask"
          ]

          # Muestra las imágenes
          axes[0, 0].imshow(original_image)
          axes[0, 0].set_title(titles[0])
          axes[0, 1].imshow(reconstructed_image)
          axes[0, 1].set_title(titles[1])
          axes[0, 2].imshow(landmark_sample, cmap='gray')
          axes[0, 2].set_title(titles[2])
          axes[0, 3].imshow(mask_sample, cmap='gray')
          axes[0, 3].set_title(titles[3])
          axes[0, 4].imshow(face_part_sample)
          axes[0, 4].set_title(titles[4])
          axes[1, 0].imshow(landmark_original, cmap='gray')
          axes[1, 0].set_title(titles[5])
          axes[1, 1].imshow(face_mask_original, cmap='gray')
          axes[1, 1].set_title(titles[5])
          axes[1, 2].imshow(face_part_original)
          axes[1, 2].set_title(titles[6])
          axes[1, 3].imshow(landmark_original - landmark_sample, cmap='hot')
          axes[1, 3].set_title(titles[7])
          axes[1, 4].imshow(mask_sample - face_mask_original, cmap='hot')
          axes[1, 4].set_title(titles[8])

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
