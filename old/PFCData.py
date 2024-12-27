from neuralgym.data import DataFromFNames
import random
import cv2
import time
import os

import tensorflow as tf
import numpy as np
import threading
from neuralgym.ops.image_ops import np_random_crop

READER_LOCK = threading.Lock()

class PFCData(DataFromFNames):
    def __init__(self, fnamelists, shapes, random=False, random_crop=False,\
                 fn_preprocess=None, dtypes=tf.float32,\
                 enqueue_size=32, queue_size=256, nthreads=16,\
                 return_fnames=False, filetype='image',\
                 preprocessed_dir = None):
        super().__init__(fnamelists, shapes, random, random_crop, fn_preprocess,
                         dtypes, enqueue_size, queue_size, nthreads,
                         return_fnames, filetype)
        if preprocessed_dir is None:
            exit('preprocessed_dir is None')
        self.preprocessed_dir = preprocessed_dir

    def read_img(self, filename):
        img = cv2.imread(filename)
        if img is None:
            print('image is None, sleep this thread for 0.1s.')
            time.sleep(0.1)
            return img, True
        if self.fn_preprocess:
            img = self.fn_preprocess(img)
        # Add preprocessed images as channels
        filename = os.path.splitext(os.path.basename(filename))[0]
        base = os.path.join(self.preprocessed_dir, filename)
        if not os.path.exists(base + '_landmarks.jpg') or  \
            not os.path.exists(base + '_mask.jpg') or \
            not os.path.exists(base + '_face_part.jpg'):
            print('landmarks, mask, or part not found for %s' % base)
            exit(1)
        else:
            landmarks = cv2.imread(base + '_landmarks.jpg', cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(base + '_mask.jpg', cv2.IMREAD_GRAYSCALE)
            part = cv2.imread(base + '_face_part.jpg')

            landmarks = np.expand_dims(landmarks, axis=-1)
            mask = np.expand_dims(mask, axis=-1)

            final_tensor = np.concatenate([img, landmarks, mask, part], axis=2)
        return final_tensor, False
    
    # def next_batch(self):
    #     batch_data = []
    #     for _ in range(self.enqueue_size):
    #         error = True
    #         while error:
    #             error = False
    #             if self.random:
    #                 filenames = random.choice(self.fnamelists_)
    #             else:
    #                 with READER_LOCK:
    #                     filenames = self.fnamelists_[self.index]
    #                     self.index = (self.index + 1) % self.file_length
    #             imgs = []
    #             random_h = None
    #             random_w = None
    #             for i in range(len(filenames)):
    #                 img, error = self.read_img(filenames[i])
    #                 if self.random_crop:
    #                     img, random_h, random_w = np_random_crop(
    #                         img, tuple(self.shapes[i][:-1]),
    #                         random_h, random_w, align=False)  # use last rand
    #                 else:
    #                     img = cv2.resize(img, tuple(self.shapes[i][:-1][::-1]))
    #                 imgs.append(img)
    #         print(imgs)
    #         if self.return_fnames:
    #             batch_data.append(imgs + list(filenames))
    #         else:
    #             batch_data.append(imgs)
    #     return zip(*batch_data)
