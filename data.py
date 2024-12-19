import tensorflow as tf
import os

def parse_test_image(original_img_path, target_size = (128, 128)):
    original_img_path_str = original_img_path.numpy().decode('utf-8')

    # Cargar imagen original (RGB)
    original_img = tf.io.read_file(original_img_path_str)
    original_img = tf.image.decode_jpeg(original_img, channels=3)
    original_img = tf.image.resize(original_img, (128, 128))
    original_img = (original_img / 255) -1.

    print(tf.shape(original_img))
    return  original_img


def parse_image_set(original_img_path, feature_img_dir, target_size = (128, 128)):
    # Nombre base de la imagen
    original_img_path_str = original_img_path.numpy().decode('utf-8')
    feature_img_dir_str = feature_img_dir.numpy().decode('utf-8')

    filename = os.path.splitext(os.path.basename(original_img_path_str))[0]
    
    # Cargar imagen original (RGB)
    original_img = tf.io.read_file(original_img_path_str)
    original_img = tf.image.decode_jpeg(original_img, channels=3)
    original_img = tf.image.resize(original_img, (128, 128))
    original_img = (original_img / 127.5) -1.

    feature_names = [('landmarks', 1), ('mask', 1), ('face_part', 3)]
    feature_imgs = []

    for name, channels in feature_names:
        feature_path = os.path.join(feature_img_dir_str, f'{filename}_{name}.jpg')
        feature_img = tf.io.read_file(feature_path)
        feature_img = tf.image.decode_jpeg(feature_img, channels=channels)
        feature_img = tf.image.resize(feature_img, target_size)
        feature_img = (feature_img / 127.5) -1.
        feature_imgs.append(feature_img)
    
    # Concatenar las imágenes
    combined_img = tf.concat([
        original_img,     # RGB
        feature_imgs[0],  # Grayscale
        feature_imgs[1],  # Grayscale
        feature_imgs[2]   # RGB
    ], axis=-1)
    
    return combined_img

def create_image_dataset(original_img_dir, feature_img_dir, batch_size=32, target_size=(128, 128)):
    # Obtener lista de rutas de imágenes originales
    original_img_paths = [
        os.path.join(original_img_dir, f) 
        for f in os.listdir(original_img_dir) 
        if f.endswith('.jpg')
    ]
    list_ds = tf.data.Dataset.list_files(str(original_img_dir+'/*'), shuffle=False)
    list_ds = list_ds.shuffle(100, reshuffle_each_iteration=False)

    AUTOTUNE = tf.data.AUTOTUNE

    list_ds = list_ds.map(
        lambda x: tf.py_function(
            func=parse_image_set, 
            inp=[x, feature_img_dir, target_size], 
            Tout=tf.float32
        ), 
        num_parallel_calls=AUTOTUNE
    )
    
    list_ds = list_ds.batch(batch_size)
    list_ds = list_ds.filter(lambda x: tf.shape(x)[0] == batch_size)

    return list_ds

def load_test_set(original_img_dir, batch_size=32, target_size=(128, 128)):
    # Nombre base de la imagen
    original_img_paths = [
        os.path.join(original_img_dir, f) 
        for f in os.listdir(original_img_dir) 
        if f.endswith('.jpg')
    ]
    list_ds = tf.data.Dataset.list_files(str(original_img_dir+'/*'), shuffle=False)
    list_ds = list_ds.shuffle(100, reshuffle_each_iteration=False)

    AUTOTUNE = tf.data.AUTOTUNE


    list_ds = list_ds.map(
        lambda x: tf.py_function(
            func=parse_test_image, 
            inp=[x, target_size], 
            Tout=tf.float32
        ), 
        num_parallel_calls=AUTOTUNE
    )

    list_ds = list_ds.batch(batch_size)
    list_ds = list_ds.filter(lambda x: tf.shape(x)[0] == batch_size)

    return list_ds