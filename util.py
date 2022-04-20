# Import Statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
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


def show_generator_results(generator_network):
    for k in range(1):
        fig = plt.figure(figsize=(15, 15))
        fake_samples = get_fake_samples(generator_network, 9, noise_size)
        fake_samples = (fake_samples+1.0)/2.0
        q = 1
        plt.subplot(990 + 1)
        plt.imshow(fake_samples[0])
        q = q + 1
        plt.axis('off')
            #plt.title(trainY[i])
    return fig
