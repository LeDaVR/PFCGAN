from data import load_test_set
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
original_img_dir = config["paths"]["test_img_dir"]
batch_size = config["hyper_parameters"]["batch_size"]

train_dataset = load_test_set(original_img_dir, batch_size=batch_size)
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


def evaluate_metrics(original_images, generated_images):
    # From -1 to 1
    original_images = (original_images + 1.) / 2.
    generated_images = (generated_images + 1.) / 2.

    # PSNR
    psnr_values = tf.image.psnr(original_images, generated_images, max_val=1.0)

    # SSIM
    ssim_values = tf.image.ssim(original_images, generated_images, max_val=1.0)

    metrics = {
        "PSNR": tf.reduce_mean(psnr_values).numpy(),
        "SSIM": tf.reduce_mean(ssim_values).numpy(),
        # "LPIPS": tf.reduce_mean(lpips_values).numpy(),
    }

    return metrics


def reparametrize(z_mean, z_log_sigma):
  eps = tf.random.normal(shape=z_mean.shape)
  return eps * tf.exp(z_log_sigma * .5) + z_mean

# Run inference with 4 random z for each image
def generate_and_save_images(predictions, original, mask, step, show=False):
    print(tf.shape(predictions))

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

def show_embedding(landmarks, mask, face_part ):
    fig = plt.figure(figsize=(10, 10))

    # Four Images, plot 3 images for each one (12 images total)
    for i in range(4):
        plt.subplot(4, 3, (i * 3) + 1)
        plt.imshow(tf.sigmoid(landmarks[i]), cmap='gray')
        plt.axis('off')
        plt.title("Landmarks")
        plt.subplot(4, 3, (i * 3) + 2)
        plt.imshow(tf.sigmoid(mask[i]), cmap='gray')
        plt.axis('off')
        plt.title("Mask")
        plt.subplot(4, 3, (i * 3) + 3)
        plt.imshow((face_part[i] + 1.) /2.)
        plt.axis('off')
        plt.title("Face Part")


        

    plt.show()
    plt.close()


def inference(lencoder, ldecoder, fencoder, mdecoder, fdecoder, generator, z, batch_incomplete, batch_mask):
    l_mu, l_log_var = lencoder([batch_incomplete, z, batch_mask], training=False)
    reparametrized_landmarks = reparametrize(l_mu, l_log_var)
    f_mu, f_log_var = fencoder([batch_incomplete, z, batch_mask], training=False)
    reparametrized_face= reparametrize(f_mu, f_log_var)
    emb = tf.concat([reparametrized_landmarks, reparametrized_face], axis=-1)
    fake = generator([emb, batch_incomplete, batch_mask], training=False)
    landmarks = ldecoder(reparametrized_landmarks, training=False)
    mask = mdecoder(emb, training=False)
    face_part = fdecoder(emb, training=False)
    return fake, landmarks, mask, face_part

for batch in train_dataset:
    num_examples_to_generate = 4
    for step, item in enumerate(batch):
        mask_batch = mask_rgb(1)
        mask_batch = tf.repeat(tf.expand_dims(mask_batch[0,:,:,:], 0), repeats=4, axis=0)
        z_seed = tf.random.normal([num_examples_to_generate, 512])
        tf.print(tf.shape(mask_batch))
        samples = []
        for i in range(4):
            samples += [item * (1. - mask_batch[i,:,:,:])]
        # create a tensor with 4 times the image
        # sample =  tf.repeat(tf.expand_dims(sample, 0), repeats=4, axis=0)
        samples = tf.stack(samples)
        print("queso", samples.shape)
        # show images and masks
        
        for i in range(4):
            plt.subplot(2, 4, i+1)
            plt.imshow((samples[i] + 1.) /2.)
            plt.axis('off')
            plt.title("Sample {}".format(i))
            plt.subplot(2, 4, i+5)
            plt.imshow(mask_batch[i])
            plt.axis('off')
            plt.title("Mask {}".format(i))
        plt.show()
        plt.close()


        predictions, landmraks, masks, face_parts = inference(pfcGan.landmark_encoder,
                                pfcGan.landmark_decoder, 
                                pfcGan.face_encoder, 
                                pfcGan.face_mask_decoder, 
                                pfcGan.face_part_decoder, 
                                pfcGan.generator, 
                                z = z_seed,
                                batch_incomplete = samples,
                                batch_mask = mask_batch[:,:,:,0:1])
        generate_and_save_images(predictions, samples, mask_batch, step, show=True)
        show_embedding(landmraks, masks, face_parts)
        print(evaluate_metrics(tf.stack([item, item , item, item]), predictions))
        print("Inference done for z={}".format(step))

