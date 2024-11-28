import tensorflow as tf

# Creamos un batch de imágenes de ejemplo
batch_images = tf.random.normal(shape=(10, 128, 128, 3))  # 10 imágenes de 128x128 con 3 canales
mascara = tf.random.uniform(shape=(128, 128, 1), minval=0, maxval=2, dtype=tf.float32)
mascara = tf.cast(mascara > 0.5, tf.float32)  # Convertir a máscara binaria

# Método 1: Multiplicación directa (broadcast)
imagenes_enmascaradas = batch_images * mascara
imagenes_enmascaradas = mascara * batch_images

# Método 2: Usando tf.expand_dims para hacer broadcast
mascara_expandida = tf.expand_dims(mascara, axis=0)  # Añade dimensión de batch
imagenes_enmascaradas_v2 = batch_images * mascara_expandida

# Verificar formas
print("Forma del batch original:", batch_images.shape)
print("Forma de la máscara:", mascara.shape)
print("Forma de las imágenes enmascaradas:", imagenes_enmascaradas.shape)