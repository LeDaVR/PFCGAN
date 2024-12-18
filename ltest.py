### Setup
import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time

from IPython import display

from data import create_image_dataset
from utils import generate_and_save_images, mask_rgb


### Load and prepare the dataset

# Hyperparameters
original_img_dir = 'D:/My Files/UNSA/PFCIII/prepro/train_set/original'
feature_img_dir = 'D:/My Files/UNSA/PFCIII/prepro/train_set/processed'
# original_img_dir = 'D:/My Files/UNSA/PFCIII/prepro/original'
# feature_img_dir = 'D:/My Files/UNSA/PFCIII/prepro/processed'
batch_size = 8
LANDMARK_RECONSTRUCTION = 2
FACE_MASK_RECONSTRUCTION = 1
FACE_PART_RECONSTRUCTION = 4
DOMAIN_KL = 30
GLOBAL_DISCRIMINATOR_LOSS = 1
CONSISTENCY_LOSS = 1
EXTRACTOR_RECONSTRUCTION = 28
GEN_LOSS = 1
EXTRACTOR_KL = 30


train_dataset = create_image_dataset(original_img_dir, feature_img_dir, batch_size=batch_size)
print(train_dataset)

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, input_channels=3, output_channels=1):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.input_channels = input_channels
    self.output_channels = output_channels
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(128, 128, input_channels)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=4, strides=(2, 2), activation='leaky_relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=4, strides=(2, 2), activation='leaky_relu'),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=4, strides=(2, 2), activation='leaky_relu'),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=4, strides=(2, 2), activation='leaky_relu'),
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=4, strides=(2, 2), activation='softplus'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=16*16*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(16, 16, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=16, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2D(
                filters=output_channels, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x, y_true):
  y_true = (y_true + 1.) / 2.
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.weighted_cross_entropy_with_logits(logits=x_logit, labels=y_true, pos_weight=tf.constant(7.))
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)

  tf.print(logpx_z)
  return x_logit , -tf.reduce_mean(logpx_z + logpz - logqz_x)

### Save checkpoints

# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                  discriminator_optimizer=discriminator_optimizer,
#                                  face_optimizer=face_optimizer,
#                                  landmark_optimizer=landmark_optimizer,
#                                  generator=generator,
#                                  discriminator=discriminator,
#                                  landmark_encoder=landmark_encoder,
#                                  landmark_decoder=landmark_decoder,
#                                  face_encoder=face_encoder,
#                                  face_mask_decoder=face_mask_decoder,
#                                  face_part_decoder=face_part_decoder,
#                                  )

## Define the training loop


EPOCHS = 100
noise_dim = 512
num_examples_to_generate = batch_size
model = CVAE(noise_dim, input_channels=3, output_channels=1)

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
    with tf.GradientTape() as landmark_tape:
      # Prepare the batch
      batch_size = tf.shape(batch)[0]
      batch_mask = mask_rgb(batch_size)
      batch_original = batch[:,:,:,0:3]
      batch_original_incomplete = batch_original * (1. - batch_mask)
      batch_landmarks = batch[:,:,:,3:4]
      batch_face_mask = batch[:,:,:,4:5]
      batch_face_part = batch[:,:,:,5:8]

      # Landkard encoder
      
      x_logit, loss = compute_loss(model, batch_original_incomplete, batch_landmarks)
    gradients = landmark_tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    tf.print("losses step", loss)

    return {
      "outputs": {
        "x_logit": x_logit,
      },
      "losses": {	
        "loss": loss,
      }
    }

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for step, image_batch in enumerate(dataset):
      values = train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)

    tf.print("losses", values["losses"])	
    # generate_and_save_images(generator,
    #                       epoch = epoch + 1,
    #                       args = seed)

    if (epoch + 1) % 4 == 0:
      outputs = values["outputs"]
      landmark_sample = tf.sigmoid(outputs["x_logit"][0])
      fig = plt.figure(figsize=(6, 6))
      plt.subplot(1, 1, 1)
      plt.imshow(landmark_sample , cmap='gray')
      plt.axis('off')
      plt.show()

    # # Save the model every 15 epochs
    # if (epoch + 1) % 15 == 0:
    #   checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  # generate_and_save_images(generator,
  #                          epoch = epochs,
  #                          args =seed)


## Train the model
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
train(train_dataset, EPOCHS)

