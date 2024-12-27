import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Usar solo la GPU 0



class SimpleGAN:
    def __init__(self, latent_dim=100, img_shape=(128, 128, 3)):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        
        # Construir generador y discriminador
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        # Modelo adversarial
        self.discriminator.trainable = False
        self.adversarial_model = self.build_adversarial_model()

    def build_generator(self):
        """Construir generador"""
        model = models.Sequential([
            layers.Dense(8*8*256, input_dim=self.latent_dim),
            layers.Reshape((8, 8, 256)),
            
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),

            layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh')
        ])
        return model

    def build_discriminator(self):
        """Construir discriminador"""
        model = models.Sequential([
            layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', 
                          input_shape=self.img_shape),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])
        return model

    def build_adversarial_model(self):
        """Construir modelo adversarial"""
        model = models.Sequential([
            self.generator,
            self.discriminator
        ])
        return model

    def train_step(self, real_images, batch_size):
        """Paso de entrenamiento"""
        # Generar ruido aleatorio
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        
        # Generar imágenes falsas
        generated_images = self.generator.predict(noise)
        
        # Entrenar discriminador
        # Combinar imágenes reales y generadas
        x_combined = np.concatenate([real_images, generated_images])
        y_combined = np.concatenate([
            np.ones((batch_size, 1)),  # Imágenes reales
            np.zeros((batch_size, 1))  # Imágenes generadas
        ])
        
        # Entrenar discriminador
        d_loss = self.discriminator.train_on_batch(x_combined, y_combined)
        
        # Entrenar generador
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        g_loss = self.adversarial_model.train_on_batch(
            noise, 
            np.ones((batch_size, 1))  # Intentar engañar al discriminador
        )
        
        return {
            "discriminator_loss": d_loss,
            "generator_loss": g_loss
        }

def load_data(dataset_path, img_shape=(128, 128, 3), batch_size=32):
    """Cargar y preprocesar dataset de imágenes"""
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        image_size=img_shape[:2],
        batch_size=batch_size,
        labels=None,
    )
    
    def preprocess(image):
        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5  # Normalizar entre -1 y 1
        return image
    
    dataset = dataset.map(preprocess)
    dataset = dataset.filter(lambda x: tf.shape(x)[0] == batch_size)
    return dataset

def main():
    # Configuración
    batch_size = 13
    epochs = 10000
    latent_dim = 100
    img_shape = (128, 128, 3)
    
    # Cargar datos
    dataset_path = 'D:/My Files/UNSA/PFCIII/prepro/original'
    dataset = load_data(dataset_path, img_shape, batch_size)    
    # Inicializar GAN
    gan = SimpleGAN(latent_dim=latent_dim, img_shape=img_shape)

    gan.generator.summary()
    gan.discriminator.summary()
    
    # Compilar modelos
    gan.discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy'
    )
    
    gan.adversarial_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
        loss='binary_crossentropy'
    )
    
    # Entrenamiento
    for epoch in range(epochs):
        for step, real_images in enumerate(dataset):
            # Realizar paso de entrenamiento
            losses = gan.train_step(real_images, batch_size)
            
            # Imprimir métricas
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}")
                print(f"Discriminator Loss: {losses['discriminator_loss']}")
                print(f"Generator Loss: {losses['generator_loss']}")
            
            # Guardar imágenes generadas
            if step % 1000 == 0:
                noise = np.random.normal(0, 1, (16, latent_dim))
                generated_images = gan.generator.predict(noise)
                
                plt.figure(figsize=(10,10))
                for i in range(16):
                    plt.subplot(4, 4, i+1)
                    img = (generated_images[i] * 127.5 + 127.5).astype(np.uint8)
                    plt.imshow(img)
                    plt.axis('off')
                plt.tight_layout()
                # plt.savefig(f'generated_images_epoch_{epoch}_step_{step}.png')
                plt.show()
                plt.close()

if __name__ == "__main__":
    main()
