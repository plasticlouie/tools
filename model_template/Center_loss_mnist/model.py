from common import *
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data/mnist', one_hot=True)

class model_common(object):
    def __init__(self):
        pass

def log(tensor, name=''):
    print name,':',tensor.get_shape(), tensor.dtype

def prelu(x, rate=0.3):
    pos_mask = tf.greater_equal(x, 0.)
    neg_mask = tf.logical_not(pos_mask)
    pos_mask = tf.cast(pos_mask, tf.float32)
    neg_mask = tf.cast(neg_mask, tf.float32)
    return x*pos_mask + (x*rate)*neg_mask


class mnist_center_loss_model(object):
    def __init__(self, config=None):
        num_classes = config.num_classes
        alpha       = config.alpha
        beta        = config.beta
        center_dim  = 2
        # lr          = 1e-2

        #np_centers = np.zeros(shape=(num_classes, center_dim),dtype=np.float32)
        #np_centers[:,0] = np.arange(num_classes)
        #centers = tf.Variable(tf.zeros(shape=(num_classes, 2),dtype=tf.float32))
        np_centers  = np.float32(np.random.rand(num_classes, center_dim))

        print np_centers.shape
        centers     = tf.Variable(np_centers)

        x = tf.placeholder(tf.float32, shape=(None,784))
        # y: (batch_size,)
        y = tf.placeholder(tf.int32, shape=(None,))
        lr = tf.placeholder(tf.float32)
        with slim.arg_scope(self.arg_scope()):
            net = tf.reshape(x, (-1,28,28,1))
            net = slim.conv2d(net, 32, [5,5])
            net = slim.conv2d(net, 32, [5,5])
            log(net, 'conv1')
            net = slim.max_pool2d(net, [2,2])
            net = slim.conv2d(net, 64, [5,5])
            net = slim.conv2d(net, 64, [5,5])
            log(net, 'conv2')
            net = slim.max_pool2d(net, [2,2])
            net = slim.conv2d(net, 128, [5,5])
            net = slim.conv2d(net, 128, [5,5])
            log(net, 'conv3')
            net = slim.max_pool2d(net, [2,2])
            net = slim.flatten(net)
            net = slim.fully_connected(net, center_dim, activation_fn=None)
            features = net
            scale = tf.norm(features,ord=2, axis=1, keep_dims=True)
            self.scale = scale

            #net = self.normalize(net)
            net = prelu(net)
        logits      = slim.fully_connected(net, num_classes, activation_fn=None)
        probs       = tf.nn.softmax(logits)
        classes     = tf.cast(tf.argmax(probs, axis=1),tf.int32)
        num_corr    = tf.reduce_sum(tf.cast(tf.equal(classes, y),tf.float32))

        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
        softmax_loss = tf.reduce_mean(softmax_loss)

        # batch_centers: (batch_size, 2)
        batch_centers   = tf.gather(centers, y)
        log(batch_centers,'batch_centers')

        center_loss     = tf.square(net-batch_centers) * 0.5
        log(center_loss,'center_loss')

        center_loss     = tf.reduce_sum(center_loss, axis=1)
        log(center_loss,'center_loss')

        center_loss     = tf.reduce_mean(center_loss)

        loss = softmax_loss + beta * center_loss

        saver = tf.train.Saver()

        optimizer = tf.train.MomentumOptimizer(lr, 0.9)
        train_op = optimizer.minimize(loss)

        self.x = x
        self.y = y
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.center_dim = center_dim
        self.net = net
        self.centers = centers
        self.batch_centers = batch_centers
        self.np_centers = np_centers
        self.logits = logits
        self.probs = probs
        self.num_corr = num_corr
        self.classes = classes
        self.center_loss = center_loss
        self.softmax_loss = softmax_loss
        self.loss = loss
        self.optimizer = optimizer
        self.train_op = train_op
        self.saver = saver

        pass

    def arg_scope(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=prelu):
            with slim.arg_scope([slim.conv2d], padding='SAME') as sc:
                return sc

    def normalize(self, tensor):
        mode = 1
        if mode == 1:
            # tensor: (batch_size, dimension)
            #scale = tf.sqrt( tf.reduce_sum( tf.square(tensor),axis=1 ) )
            #scale = tf.expand_dims(scale, axis=-1)
            #print tensor.get_shape()
            scale = tf.norm(tensor,ord=2, axis=1, keep_dims=True) + 0.001
            #pos_mask = tf.greater(scale, 0.001)
            #neg_mask = tf.logical_not(pos_mask)
            #scale = scale * pos_mask +
            tensor = tensor / scale
        else:
            mean = tf.reduce_mean(tensor, axis=1)
            mean = tf.expand_dims(mean, axis=-1)
            std = tf.sqrt(tf.reduce_mean( tf.square(tensor-mean),axis=1 ))
            std = tf.expand_dims(std, axis=-1)
            tensor = (tensor-mean)/std

        return tensor

    def get_scale(self, sess, batch_images):
        feed_dict = {self.x: batch_images}
        return sess.run(self.scale, feed_dict=feed_dict)

    def train_on_batch(self, sess, batch_images, batch_labels, lr=1e-2):
        feed_dict = {self.x: batch_images, self.y:batch_labels, self.lr:lr}
        run_list = [self.train_op, self.loss, self.softmax_loss, self.center_loss, self.net, self.batch_centers]
        [_, loss, cls_loss, center_loss, net, batch_centers] = sess.run(run_list, feed_dict=feed_dict)

        centers_delta = np.zeros(shape=(self.num_classes,self.center_dim))
        center_diff = batch_centers - net

        for i in range(self.num_classes):
            class_mask = (batch_labels==i)
            if np.any(class_mask):
                centers_delta[i] = np.mean(center_diff[class_mask], axis=0)

        self.np_centers = self.np_centers - self.alpha * centers_delta
        sess.run( self.centers.assign(self.np_centers) )
        result_dict = {'loss':loss, 'softmax_loss':cls_loss, 'center_loss':center_loss,
                        'centers_delta':centers_delta}
        return result_dict

    def train_on_generator(self, sess, data_generator, num_batches=None):
        loss = []
        cls_loss = []
        center_loss = []

        current_index = 0
        for batch_images, batch_labels in data_generator:

            if num_batches is not None:
                current_index += 1
                if current_index > num_batches:
                    break
            result = self.train_on_batch(sess, batch_images, batch_labels)
            loss = loss + [result['loss']]
            cls_loss = cls_loss + [result['softmax_loss']]
            center_loss = center_loss + [result['center_loss']]
        return loss, cls_loss, center_loss

    def inference_on_batch(self, sess, batch_images):
        feed_dict = {self.x:batch_images}
        [classes] = sess.run([self.classes], feed_dict=feed_dict)
        return classes

    def compute_features_on_batch(self, sess, batch_images, need_labels=False):
        feed_dict = {self.x:batch_images}
        [batch_features] = sess.run([self.net], feed_dict=feed_dict)
        if need_labels:
            return batch_features, batch_labels
        else:
            return batch_features

    def compute_features_on_generator(self, sess, data_generator, num_batches=None, need_labels=True):
        current_index = 0
        features = []
        labels = []
        for batch_images, batch_labels in data_generator:

            if num_batches is not None:
                current_index += 1
                if current_index > num_batches:
                    break

            batch_features = self.compute_features_on_batch(sess, batch_images)
            features = features + [batch_features]

            if need_labels:
                labels = labels + [batch_labels]

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        if need_labels:
            return features, labels
        else:
            return features

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


class mnist_softmax_loss_model(object):
    def __init__(self):
        model_name = 'mnist_softmax'
        num_classes = 10
        lr          = 1e-2
        center_dim  = 2
        with tf.variable_scope(model_name):
            x = tf.placeholder(tf.float32, shape=(None,784))
            y = tf.placeholder(tf.int32, shape=(None,))
            with slim.arg_scope(self.arg_scope()):
                net = tf.reshape(x, (-1,28,28,1))
                net = slim.conv2d(net, 32, [5,5])
                net = slim.conv2d(net, 32, [5,5])
                net = slim.max_pool2d(net, [2,2])
                net = slim.conv2d(net, 64, [5,5])
                net = slim.conv2d(net, 64, [5,5])
                net = slim.max_pool2d(net, [2,2])
                net = slim.conv2d(net, 128, [5,5])
                net = slim.conv2d(net, 128, [5,5])
                net = slim.max_pool2d(net, [2,2])
                net = slim.flatten(net)
                net = slim.fully_connected(net, center_dim, activation_fn=None)
                features = net
                net = prelu(net)
                logits      = slim.fully_connected(net, num_classes, activation_fn=None)
                probs       = tf.nn.softmax(logits)
                classes     = tf.cast(tf.argmax(probs, axis=1),tf.int32)
                num_corr    = tf.reduce_sum(tf.cast(tf.equal(classes, y),tf.float32))

                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
                loss = tf.reduce_mean(loss)

                variables = slim.get_variables_to_restore(include=[model_name])
                saver = tf.train.Saver(variables)

                optimizer = tf.train.MomentumOptimizer(lr, 0.9)
                train_op = optimizer.minimize(loss)

        self.x = x
        self.y = y
        self.num_classes = num_classes
        self.center_dim = center_dim
        self.net = net
        self.logits = logits
        self.probs = probs
        self.num_corr = num_corr
        self.classes = classes
        self.loss = loss
        self.optimizer = optimizer
        self.train_op = train_op
        self.variables = variables
        self.saver = saver

        pass

    def arg_scope(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=prelu):
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

    def inference_on_batch(self, sess, batch_images):
        feed_dict = {self.x:batch_images}
        [classes] = sess.run([self.classes], feed_dict=feed_dict)
        return classes

    def compute_features_on_batch(self, sess, batch_images, need_labels=False):
        feed_dict = {self.x:batch_images}
        [batch_features] = sess.run([self.net], feed_dict=feed_dict)
        if need_labels:
            return batch_features, batch_labels
        else:
            return batch_features

    def compute_features_on_generator(self, sess, data_generator, num_batches=None, need_labels=True):
        current_index = 0
        features = []
        labels = []
        for batch_images, batch_labels in data_generator:

            if num_batches is not None:
                current_index += 1
                if current_index > num_batches:
                    break

            batch_features = self.compute_features_on_batch(sess, batch_images)
            features = features + [batch_features]

            if need_labels:
                labels = labels + [batch_labels]

        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)

        if need_labels:
            return features, labels
        else:
            return features

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
