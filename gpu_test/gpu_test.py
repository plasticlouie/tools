import os
import time
from time import time as tic
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data', one_hot=False)


#
# Data generator
#
class mnist_data(object):
    def __init__(self, data_path='/data/mnist', one_hot=False):
        mnist = input_data.read_data_sets(data_path, one_hot=one_hot)
        self.mnist = mnist

    def mnist_generator(self,batch_size, all_images, all_labels):
        num_cases = all_labels.shape[0]
        num_batches = np.int32( np.ceil(np.float32(num_cases)/batch_size) )
        indexes = []
        for i in range(num_batches):
            indexes = indexes + [np.arange(i*batch_size,(i+1)*batch_size)]
        return ( (all_images[patch], all_labels[patch]) for patch in indexes )

    def val_generator(self,batch_size=50):
        return self.mnist_generator(batch_size = batch_size,
                               all_images = self.mnist.validation.images,
                               all_labels = self.mnist.validation.labels)

    def train_generator(self,batch_size=50):
        return self.mnist_generator(batch_size = batch_size,
                               all_images = self.mnist.train.images,
                               all_labels = self.mnist.train.labels)

    def test_generator(self,batch_size=50):
        return self.mnist_generator(batch_size = batch_size,
                               all_images = self.mnist.test.images,
                               all_labels = self.mnist.test.labels)



#
# Model
#
class mnist_net(object):
    def __init__(self):
        num_classes = 10
        lr          = 1e-2

        x = tf.placeholder(tf.float32, shape=(None,784))
        y = tf.placeholder(tf.int32, shape=(None,))
        with slim.arg_scope(self.arg_scope()):
            net = tf.reshape(x, (-1,28,28,1))
            net = slim.conv2d(net, 32, [5,5])
            net = slim.max_pool2d(net, [2,2])
            net = slim.conv2d(net, 64, [5,5])
            net = slim.max_pool2d(net, [2,2])
            net = slim.flatten(net)
            net = slim.fully_connected(net, 512)
        logits      = slim.fully_connected(net, num_classes, activation_fn=None)
        probs       = tf.nn.softmax(logits)
        classes     = tf.cast(tf.argmax(probs, axis=1),tf.int32)
        num_corr    = tf.reduce_sum(tf.cast(tf.equal(classes, y),tf.float32))

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_mean(loss)

        optimizer = tf.train.MomentumOptimizer(lr, 0.9)
        train_op = optimizer.minimize(loss)

        self.x = x
        self.y = y
        self.num_classes = num_classes
        self.net = net
        self.logits = logits
        self.probs = probs
        self.num_corr = num_corr
        self.classes = classes
        self.loss = loss
        self.optimizer = optimizer
        self.train_op = train_op


    def arg_scope(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu):
            with slim.arg_scope([slim.conv2d], padding='SAME') as sc:
                return sc

    def train_on_batch(self, sess, batch_images, batch_labels):
        feed_dict = {self.x: batch_images, self.y:batch_labels}
        [_, loss] = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def train_on_generator(self, sess, data_generator, num_batches=None):
        loss = []

        current_index = 0
        for batch_images, batch_labels in data_generator:

            if num_batches is not None:
                current_index += 1
                if current_index > num_batches:
                    break

            temp_loss = self.train_on_batch(sess, batch_images, batch_labels)
            loss = loss + [temp_loss]
        return loss

    def test_acc_on_generator(self, sess, data_generator, num_batches=None):
        current_index = 0
        num_cases = 0.
        num_corrs = 0.
        for batch_images, batch_labels in data_generator:

            if num_batches is not None:
                current_index += 1
                if current_index > num_batches:
                    break

            feed_dict = {self.x:batch_images, self.y: batch_labels}
            num_corrs += sess.run(self.num_corr, feed_dict=feed_dict)
            num_cases += batch_labels.shape[0]
        acc = num_corrs / num_cases
        return acc

#
# Training
#
num_epochs = 5
net = mnist_net()

mnist = mnist_data()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

val_generator = mnist.val_generator()

t1 = tic()
for e in range(num_epochs):
    train_generator = mnist.train_generator()
    loss = net.train_on_generator(sess, train_generator)
    print('Epoch: {:d}/{:d},  Loss:{:4f}'.format(e+1,num_epochs,loss[-1]))
t2 = tic()
sec = t2 - t1

val_acc = net.test_acc_on_generator(sess, val_generator)

print("---------------------------")
print("Train for {} epochs (5w images per epoch)".format(num_epochs))
print("")
print("val accuracy  : {:4f}".format(val_acc))
print("training time : {:4f} secs".format(sec))
print("secs/epoch    : {:4f}".format(sec/num_epochs))
print("normal        : {:d} secs/epoch in GPU mode".format(5))
