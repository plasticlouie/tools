import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data


class mnist_data(object):
    def __init__(self, data_path='/data/mnist', one_hot=False):
        mnist = input_data.read_data_sets(data_path, one_hot=one_hot)
        self.mnist = mnist

    def mnist_generator(self,batch_size, all_images, all_labels, shuffle=True):
        num_cases   = all_labels.shape[0]
        num_batches = np.int32( np.ceil(np.float32(num_cases)/batch_size) )
        if shuffle:
            image_indxes = np.random.permutation(num_cases)
        else:
            image_indxes = np.arange(num_cases)

        indexes = []
        for i in range(num_batches):
            indexes = indexes + [ image_indxes[ i*batch_size:(i+1)*batch_size ] ]
            #indexes = indexes + [np.arange(i*batch_size,(i+1)*batch_size)]
        return ( (all_images[patch], all_labels[patch]) for patch in indexes )

    def val_generator(self, batch_size=50, shuffle=True):
        return self.mnist_generator(batch_size = batch_size,
                                    all_images = self.mnist.validation.images,
                                    all_labels = self.mnist.validation.labels,
                                    shuffle    = shuffle)

    def train_generator(self,batch_size=50, shuffle=True):
        return self.mnist_generator(batch_size = batch_size,
                                    all_images = self.mnist.train.images,
                                    all_labels = self.mnist.train.labels,
                                    shuffle    = shuffle)

    def test_generator(self,batch_size=50, shuffle=True):
        return self.mnist_generator(batch_size = batch_size,
                                    all_images = self.mnist.test.images,
                                    all_labels = self.mnist.test.labels,
                                    shuffle    = shuffle)

color_map = [(220, 20, 60),
    ( 70,130,180),
    (111, 74,  0),
    ( 81,  0, 81),
    (128, 64,128),
    (244, 35,232),
    (250,170,160),
    (230,150,140),
    ( 70, 70, 70),
    (102,102,156),
    (190,153,153),
    (180,165,180),
    (150,100,100),
    (150,120, 90),
    (153,153,153),
    (153,153,153),
    (250,170, 30),
    (220,220,  0),
    (107,142, 35),
    (152,251,152),
     (255,  0,  0)]
color_map = np.asarray(color_map) / 255.
