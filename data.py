import tensorflow as tf
import os
import glob

class MultiChannelDataLoader:
    def __init__(self, original_dir, preprocessed_dir, img_size=(128, 128)):
        self.original_dir = original_dir
        self.preprocessed_dir = preprocessed_dir
        self.img_size = img_size
        
    def load_and_preprocess_image(self, image_path, num_channels=3):
        # Leer la imagen
        img = tf.io.read_file(image_path)
        # Decodificar la imagen
        img = tf.image.decode_png(img, channels=num_channels)
        # Convertir a float32 y normalizar
        img = tf.divide(tf.cast(img, tf.float32), 127.5) - 1.
        # Redimensionar
        img = tf.image.resize(img, self.img_size)
        return img

    def create_dataset(self, batch_size=32):
        original_files = sorted(glob.glob(os.path.join(self.original_dir, "*.jpg")))
        landmarks_files = sorted(glob.glob(os.path.join(self.preprocessed_dir, "*_landmarks.jpg")))
        face_mask_files = sorted(glob.glob(os.path.join(self.preprocessed_dir, "*_mask.jpg")))
        face_part_files = sorted(glob.glob(os.path.join(self.preprocessed_dir, "*_face_part.jpg")))

        tf.print("Found {} images in original directory".format(len(original_files)))
        tf.print("Found {} images in landmarks directory".format(len(landmarks_files)))
        tf.print("Found {} images in face_mask directory".format(len(face_mask_files)))
        tf.print("Found {} images in face_part directory".format(len(face_part_files)))
        
        # Ensure all are the same number of images
        if len(original_files) != len(face_mask_files):
            raise ValueError("El número de imágenes en ambos directorios debe ser igual")
        if len(original_files) != len(face_part_files):
            raise ValueError("El número de imágenes en ambos directorios debe ser igual")
        if len(original_files) != len(landmarks_files):
            raise ValueError("El número de imágenes en ambos directorios debe ser igual")
        
        def load_image_tuple(orig_path, landmarks_path, face_mask_path, face_part_path):
            orig_img = self.load_and_preprocess_image(orig_path, num_channels=3)
            landmark_img = self.load_and_preprocess_image(landmarks_path, num_channels=1)
            face_mask_img = self.load_and_preprocess_image(face_mask_path, num_channels=1)
            face_part_img = self.load_and_preprocess_image(face_part_path, num_channels=3)

            combined = tf.concat([orig_img, landmark_img, face_mask_img, face_part_img], axis=-1)
            return combined

        # Crear datasets de paths
        orig_ds = tf.data.Dataset.from_tensor_slices(original_files)
        landmarks_ds = tf.data.Dataset.from_tensor_slices(landmarks_files)
        face_mask_ds = tf.data.Dataset.from_tensor_slices(face_mask_files)
        face_part_ds = tf.data.Dataset.from_tensor_slices(face_part_files)
        
        # Combinar los datasets
        dataset = tf.data.Dataset.zip((orig_ds, landmarks_ds, face_mask_ds, face_part_ds))
        
        # Mapear la función de carga
        dataset = dataset.map(
            lambda x, l, m, p: tf.py_function(
                func=load_image_tuple,
                inp=[x, l, m, p],
                Tout=tf.float32
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Configurar el dataset
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

def parse_test_image(original_img_path, target_size = (128, 128)):
    original_img_path_str = original_img_path.numpy().decode('utf-8')

    # Cargar imagen original (RGB)
    original_img = tf.io.read_file(original_img_path_str)
    original_img = tf.image.decode_jpeg(original_img, channels=3)
    original_img = tf.image.resize(original_img, target_size)
    original_img = (tf.cast(original_img, tf.float32) / 127.5) -1.

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
    original_img = tf.image.resize(original_img, target_size)
    original_img = (tf.cast(original_img, tf.float32) / 127.5) -1.

    feature_names = [('landmarks', 1), ('mask', 1), ('face_part', 3)]
    feature_imgs = []

    for name, channels in feature_names:
        feature_path = os.path.join(feature_img_dir_str, f'{filename}_{name}.jpg')
        feature_img = tf.io.read_file(feature_path)
        feature_img = tf.image.decode_jpeg(feature_img, channels=channels)
        feature_img = tf.image.resize(feature_img, target_size)
        feature_img = (tf.cast(feature_img, tf.float32) / 127.5) -1.
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
