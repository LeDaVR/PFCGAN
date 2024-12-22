import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def generate_random_mask(image_height, image_width):
    """
    Genera una máscara aleatoria para una imagen de dimensiones dadas.

    Args:
        image_height (int): Altura de la imagen.
        image_width (int): Ancho de la imagen.

    Returns:
        tf.Tensor: Máscara binaria de forma (image_height, image_width).
    """
    minval = tf.constant(0.2, dtype=tf.float32)
    maxval = tf.constant(0.6, dtype=tf.float32)
    # Porcentaje aleatorio de la imagen que cubrirá la máscara
    mask_height_pct = tf.random.uniform([], minval=minval, maxval=maxval, dtype=tf.float32)
    mask_width_pct = tf.random.uniform([], minval=minval, maxval=maxval, dtype=tf.float32)

    # Tamaño de la máscara en píxeles
    mask_height = tf.cast(mask_height_pct * image_height, tf.int32)
    mask_width = tf.cast(mask_width_pct * image_width, tf.int32)

    # Desplazamiento aleatorio dentro del rango permitido
    max_shift_x = (image_width - mask_width) // 8
    max_shift_y = (image_height - mask_height) // 8

    shift_x = tf.random.uniform([], -max_shift_x, max_shift_x, dtype=tf.int32)
    shift_y = tf.random.uniform([], -max_shift_y, max_shift_y, dtype=tf.int32)

    two = tf.constant(2, dtype=tf.int32)

    # Coordenadas del centro de la imagen
    center_x = tf.constant(image_width // two, dtype=tf.int32)
    center_y = tf.constant(image_height // two, dtype=tf.int32)

    min_w_space = tf.constant(image_width // 8, dtype=tf.int32)
    min_h_space = tf.constant(image_height // 8, dtype=tf.int32)
    max_w_val = tf.constant( 7 * image_width // 8, dtype=tf.int32)
    max_h_val = tf.constant( 7 * image_height // 8, dtype=tf.int32)

    # Coordenadas iniciales y finales de la máscara
    semiwidth = tf.constant(mask_width // two, dtype=tf.int32)
    semiheight = tf.constant(mask_height // two, dtype=tf.int32)
    x_start = tf.clip_by_value(center_x - semiwidth + shift_x, min_w_space, max_w_val)
    y_start = tf.clip_by_value(center_y - semiheight + shift_y, min_h_space, max_h_val)

    x_end = tf.clip_by_value(x_start + mask_width, (image_width //8) , max_w_val)
    y_end = tf.clip_by_value(y_start + mask_height, (image_height // 8) , max_h_val)

    # Crear la máscara
    mask = tf.zeros((image_height, image_width), dtype=tf.float32)

    # Llenar la región seleccionada con unos
    mask = tf.tensor_scatter_nd_update(
        mask,
        indices=tf.reshape(tf.stack(tf.meshgrid(tf.range(y_start, y_end), tf.range(x_start, x_end), indexing='ij'), axis=-1), (-1, 2)),
        updates=tf.ones(((y_end - y_start) * (x_end - x_start),), dtype=tf.float32)
    )
    mask = tf.expand_dims(mask, axis=-1)

    return mask

def create_batch_mask(size, image_size=[128, 128]):
    masks = [generate_random_mask(image_size[0], image_size[1]) for _ in range(size)]
    return tf.stack(masks)

def mask_rgb(batch_size):
    image_size = [128, 128]
    
    # Genera máscaras en escala de grises
    gray_mask = create_batch_mask(batch_size, image_size)

    # Convierte a máscara RGB
    rgb_mask = tf.repeat(gray_mask, repeats=3, axis=-1)
    
    return rgb_mask


def generate_and_save_images(predictions, original, mask, step, show=False):

    fig = plt.figure(figsize=(6, 6))

    for i in range(tf.shape(predictions)[0]):
        img = (predictions[i] * mask[i]) + (original[i] * (1. - mask[i]))
        plt.subplot(4, 4, i+1)
        plt.imshow((img +1.) /2.)
        plt.axis('off')

    plt.savefig('res/image_at_step_{:04d}.png'.format(step))
    if show:
        plt.show()
    plt.close()
