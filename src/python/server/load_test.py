from numpy import random
import numpy as np
import tensorflow as tf


mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

NUM_EXAMPLES_PER_USER = 2000
BATCH_SIZE = 100
USERS = 10




def load(batch_size = BATCH_SIZE):

    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
    test_data = __format_data(batch_size, mnist_test)
    return test_data


def __format_data(batch_size, source):
    """
    从数据集source中随机抽取num个数的数据样本进行打包，每个batch的大小为batch_size
    :param batch_size:每个batch的大小
    :param source:源数据集
    :return: output_sequence:生成的数据集
    """
    all_samples = []
    output_sequence = []
    for _ in range(len(source[0])):
        all_samples.append(_)
    for i in range(0, min(len(all_samples), NUM_EXAMPLES_PER_USER), batch_size):
        batch_samples = all_samples[i:i + batch_size]
        output_sequence.append({
            'x':
                np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                         dtype=np.float32),
            'y':
                np.array([source[1][i] for i in batch_samples], dtype=np.int32)
        })
    return output_sequence
