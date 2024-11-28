import tensorflow as tf
import matplotlib.pyplot as plt

def create_inpainting_mask(image_size=(128, 128)):
    # Crear tensor de la mitad del tamaño
    mask_center = tf.ones((image_size[0]//2, image_size[1]//2), dtype=tf.float32)
    
    # Aplicar padding para llevarlo al tamaño original
    paddings = tf.constant([
        [image_size[0]//4, image_size[0]//4],  # Padding vertical
        [image_size[1]//4, image_size[1]//4]   # Padding horizontal
    ])
    
    # Aplicar padding
    out = tf.pad(mask_center, paddings, mode='CONSTANT', constant_values=0)
    
    return tf.expand_dims(out, -1)

def create_batch_mask(size):
    new_mask = create_inpainting_mask()
    batch_mask =  tf.expand_dims(new_mask, 0)
    batch_mask = tf.tile(batch_mask, [size, 1, 1, 1])
    return batch_mask

def mask_rgb(batch_size):
    image_size = (128, 128)
    
    # Crear máscara central
    mask_center = tf.zeros(image_size, dtype=tf.float32)
    mask_center = tf.tensor_scatter_nd_update(
        mask_center, 
        [[image_size[0]//4 + i, image_size[1]//4 + j] 
         for i in range(image_size[0]//2) 
         for j in range(image_size[1]//2)], 
        tf.ones((image_size[0]//2 * image_size[1]//2))
    )
    
    # Asegurar dimensiones correctas
    gray_mask = tf.reshape(mask_center, [image_size[0], image_size[1], 1])
    
    # Expandir a máscara RGB para todo el batch
    rgb_mask = tf.tile(tf.expand_dims(gray_mask, 0), [batch_size, 1, 1, 3])
    
    return rgb_mask

def generate_and_save_images(model, args, epoch, show=False):
    predictions = model(args, training=False)

    fig = plt.figure(figsize=(6, 6))

    for i in range(tf.shape(predictions)[0]):
        img = predictions[i] 
        plt.subplot(4, 4, i+1)
        plt.imshow((img +1.) /2.)
        plt.axis('off')

    plt.savefig('res/image_at_epoch_{:04d}.png'.format(epoch))
    if show:
        plt.show()
    plt.close()