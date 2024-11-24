# Migrate PIIGAN from TF1 and neuralgym to TF2.0 with keras

import tensorflow as tf
from tensorflow import keras
import logging
import numpy as np

logger = logging.getLogger()

class PIIGANModel(keras.Model):
    def __init__(self):
        super(PIIGANModel, self).__init__()
        self.cnum = 32
        
        # Inicializar capas del generador
        self.gen_conv_layers = {}
        self.gen_deconv_layers = {}
        
        # Inicializar capas del discriminador
        self.disc_local_layers = {}
        self.disc_global_layers = {}
        
        # Inicializar capas del extractor
        self.extractor_layers = {}
        
        self._build_layers()

    def _build_layers(self):
        # Conv layers
        cnum = self.cnum

        self.gen_conv_layers.update({
            'conv1': keras.layers.Conv2D(cnum, 5, 1, padding='same'),
            'conv2': keras.layers.Conv2D(2*cnum, 3, 2, padding='same'),
            'conv3': keras.layers.Conv2D(2*cnum, 3, 1, padding='same'),
            'conv4': keras.layers.Conv2D(4*cnum, 3, 2, padding='same'),
        })