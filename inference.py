from data import create_image_dataset
import matplotlib.pyplot as plt
from utils import mask_rgb
import tensorflow as tf
import os

from model import make_extractor_model, make_generator_model, make_discriminator_model, \
    make_landmark_encoder, make_landmark_decoder, \
    make_face_encoder, make_face_mask_decoder, make_face_part_decoder, make_local_discriminator

import yaml

# Cargar configuración desde el archivo YAML
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Use train data temporarily
original_img_dir = config["paths"]["original_img_dir"]
feature_img_dir = config["paths"]["feature_img_dir"]
batch_size = config["hyper_parameters"]["batch_size"]

train_dataset = create_image_dataset(original_img_dir, feature_img_dir, batch_size=batch_size)
print(train_dataset)

class PFCGAN():
   def __init__(self, landmark_encoder, landmark_decoder, face_encoder, face_mask_decoder, face_part_decoder, generator):
      self.landmark_encoder = landmark_encoder
      self.landmark_decoder = landmark_decoder
      self.face_encoder = face_encoder
      self.face_mask_decoder = face_mask_decoder
      self.face_part_decoder = face_part_decoder
      self.generator = generator
# Load the model

generator = make_generator_model()
generator.trainable = False
landmark_encoder = make_landmark_encoder()
landmark_decoder = make_landmark_decoder()
face_encoder = make_face_encoder()
face_mask_decoder = make_face_mask_decoder()
face_part_decoder = make_face_part_decoder()

pfcGan = PFCGAN(landmark_encoder, landmark_decoder, face_encoder, face_mask_decoder, face_part_decoder, generator)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
                                # generator_optimizer=generator_optimizer,
                                #  extractor_optimizer=extractor_optimizer,
                                 generator=generator,
                                 landmark_encoder=landmark_encoder,
                                 landmark_decoder=landmark_decoder,
                                 face_encoder=face_encoder,
                                 face_mask_decoder=face_mask_decoder,
                                 face_part_decoder=face_part_decoder,
                                 )
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def reparametrize(z_mean, z_log_sigma):
  eps = tf.random.normal(shape=z_mean.shape)
  return eps * tf.exp(z_log_sigma * .5) + z_mean

# Run inference with 4 random z for each image
def generate_and_save_images(predictions, original, mask, step, show=False):

    fig = plt.figure(figsize=(10, 10))

    for i in range(tf.shape(predictions)[0]):
        img = (predictions[i] * mask[i]) + (original[i] * (1. - mask[i]))
        plt.subplot(1, 4, i+1)
        plt.imshow((img +1.) /2.)
        plt.axis('off')

    plt.savefig('res/inference_at_step_{:04d}.png'.format(step))
    if show:
        plt.show()
    plt.close()

def inference(lencoder, ldecoder, fencoder, mdecoder, fdecoder, generator, z, batch_incomplete, batch_mask):
    l_mu, l_log_var = lencoder([batch_incomplete, z, batch_mask], training=False)
    reparametrized_landmarks = reparametrize(l_mu, l_log_var)
    f_mu, f_log_var = fencoder([batch_incomplete, z, batch_mask], training=False)
    reparametrized_face= reparametrize(f_mu, f_log_var)
    emb = tf.concat([reparametrized_landmarks, reparametrized_face], axis=-1)
    fake = generator([emb, batch_incomplete, batch_mask], training=False)
    return fake

for batch in train_dataset:
    num_examples_to_generate = 4
    mask_batch = mask_rgb(num_examples_to_generate)
    for step, item in enumerate(batch):
        z_seed = tf.random.normal([num_examples_to_generate, 512])
        tf.print(tf.shape(mask_batch))
        sample = item[:,:,0:3] * (1. - mask_batch[0,:,:,:])
        # create a tensor with 4 times the image
        sample =  tf.repeat(tf.expand_dims(sample, 0), repeats=4, axis=0)
        print("queso", sample.shape)
        predictions = inference(pfcGan.landmark_encoder, 
                                pfcGan.landmark_decoder, 
                                pfcGan.face_encoder, 
                                pfcGan.face_mask_decoder, 
                                pfcGan.face_part_decoder, 
                                pfcGan.generator, 
                                z = z_seed,
                                batch_incomplete = sample,
                                batch_mask = mask_batch[:,:,:,0:1])
        generate_and_save_images(predictions, sample, mask_batch, step, show=True)
        print("Inference done for z={}".format(step))

