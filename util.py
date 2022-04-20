# Import Statements
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model, load_model, save_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, BatchNormalization, Concatenate, Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, LayerNormalization, LeakyReLU, ReLU
from tensorflow.keras.layers import Reshape, UpSampling2D
from tensorflow.keras.utils import plot_model, image_dataset_from_directory
from tensorflow.keras.preprocessing.image import array_to_img, load_img, img_to_array, ImageDataGenerator

import tensorflow_addons as tfa
from tensorflow_addons.layers import SpectralNormalization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import glob
import pathlib
import cv2
import pickle
import Augmentor
from functools import partial
import string
import random
import os

#indices = [i for i in range(0, len(images))]
noise_size = 100

def get_random_noise(batch_size, noise_size):
    random_values = np.random.randn(batch_size*noise_size)
    random_noise_batch = np.reshape(random_values, (batch_size, noise_size))
    return random_noise_batch

def get_fake_samples(generator_network, batch_size, noise_size):
    random_noise_batch = get_random_noise(batch_size, noise_size)
    fake_samples = generator_network.predict_on_batch(random_noise_batch)
    return fake_samples

def get_real_samples(batch_size):
    random_indices = np.random.choice(indices, size=batch_size)
    real_images = images[np.array(random_indices),:]
    return real_images

#Name Randomization
def id_generator(size=10, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def show_generator_results(generator_network):
    for k in range(1):
        fig = plt.figure(figsize=(224, 224))
        fake_samples = get_fake_samples(generator_network, 9, noise_size)
        fake_samples = (fake_samples+1.0)/2.0
        q = 1
        plt.subplot(990 + 1)
        plt.imshow(fake_samples[0])
        randomsequence = id_generator()
            #outfile = 'FakePengu/%s.jpg' % (randomsequence)
            #plt.imsave(outfile, fake_samples[j])
        q = q + 1
        plt.axis('off')
            #plt.title(trainY[i])
    return fig
