import collections
from numpy import random

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

NUM_EXAMPLES_PER_USER = 2000
BATCH_SIZE = 100
USERS = 10




def load(num, batch_size = BATCH_SIZE):

    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
    train_data = __format_data(num, batch_size, mnist_train)
    test_data = __format_data(num//10, batch_size, mnist_test)
    return train_data, test_data

def __format_data(num, batch_size, source):
    all_samples = []
    output_sequence = []
    for _ in range(num):
        f = random.randint(0, len(source[0]))
        all_samples.append(f)
    for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x':
                np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                         dtype=np.float32),
            'y':
                np.array([source[1][i] for i in batch_samples], dtype=np.int32)
        })
    return output_sequence

def sep_data(source, num, count):
    output_sequence = []
    length = len(source[0])
    per_l = length//count
    all_samples = [i for i in range(num*per_l,(num+1)*per_l-1)]
    for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x':
                np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                         dtype=np.float32),
            'y':
                np.array([source[1][i] for i in batch_samples], dtype=np.int32)
        })
    return output_sequence

