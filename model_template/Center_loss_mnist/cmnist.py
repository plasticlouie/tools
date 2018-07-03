import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from mnist_funcs import *

#from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/data/mnist', one_hot=True)



# Learning rate
lr = 1e-4
# Max iteration number
max_iter = 2500
# Batch size during training
batch_size = 100

# Load images and labels
imgs_60k, _ = loadImageSet('./data/train-images.idx3-ubyte')
labels_60k, _ = loadLabelSet('./data/train-labels.idx1-ubyte')
imgs_10k, _ = loadImageSet('./data/t10k-images.idx3-ubyte')
labels_10k, _ = loadLabelSet('./data/t10k-labels.idx1-ubyte')

# Training set
train_imgs = np.reshape( imgs_60k[0:train_size,:], newshape=(train_size,28,28) )
train_imgs.shape = (train_size,28,28,1)
train_labels = labels_60k[0:train_size]

# Testing set
test_imgs = np.reshape( imgs_10k, newshape=(10000,28,28) )
test_imgs.shape = (10000,28,28,1)
test_labels = labels_10k


# -----------------------------------
# Build net
# -----------------------------------
inputs = tf.placeholder(tf.float32, shape=(None,28,28,1))
y = tf.placeholder(tf.int64, shape=(None,))
with slim.arg_scope([slim.conv2d, slim.fully_connected],
                    activation_fn = tf.nn.relu):
    # Convolution
    net = slim.conv2d(inputs, 32, [5,5], padding='SAME', scope='conv1')
    # Max pooling
    net = slim.max_pool2d(net, [2,2], scope = 'pool1')
    # Convolution
    net = slim.conv2d(net, 64, [5,5], padding='SAME', scope='conv2')
    # Max pooling
    net = slim.max_pool2d(net, [2,2], scope = 'pool2')
    net = slim.flatten(net)
    # Fully connected
    net = slim.fully_connected(net,1024, scope='fc1', activation_fn=tf.nn.relu)
    # Output layer
    net = slim.fully_connected(net,10, scope='fc2', activation_fn=None)
    # Softmax loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net,labels=y)
    loss = tf.reduce_mean(loss)
    # Adam optimizer
    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss)
    # Correction mask
    corr = tf.equal( tf.argmax(net,1), y )
    # Accuracy
    acc = tf.reduce_mean(tf.cast(corr, tf.float32))



# -----------------------------------
# Initialization
# -----------------------------------
sess = tf.Session()
sess.run(tf.global_variables_initializer())



# -----------------------------------
# Training
# -----------------------------------
for iter in range(max_iter):
    batch_idx = iter % (train_size/batch_size)
    batch_imgs = train_imgs[batch_idx*batch_size:(batch_idx+1)*batch_size,:,:,:]
    batch_labels = train_labels[batch_idx*batch_size:(batch_idx+1)*batch_size]
    feed_dict = {inputs:batch_imgs, y:batch_labels}
    [_,loss_value] = sess.run([train_op, loss], feed_dict=feed_dict)
    if iter % 100 == 0:
        [train_acc] = sess.run([acc], feed_dict=feed_dict)
        print 'Iter:%d, Train acc:%.2f' % (iter,train_acc*100)
        print 'Loss:',loss_value



# -----------------------------------
# Testing
# -----------------------------------
# Number of correct predictions
corr_num = 0
test_loop_num = np.int( np.ceil(10000.0/batch_size) )
for iter in range(test_loop_num):
    batch_idx = iter % (train_size/batch_size)
    batch_imgs = test_imgs[iter*batch_size:(iter+1)*batch_size,:,:,:]
    batch_labels = test_labels[iter*batch_size:(iter+1)*batch_size]
    local_batch_size = batch_imgs.shape[0]
    feed_dict = {inputs:batch_imgs, y:batch_labels}
    [test_batch_acc] = sess.run([acc], feed_dict=feed_dict)
    corr_num = corr_num + test_batch_acc*local_batch_size



# -----------------------------------
# Evaluation
# -----------------------------------
test_acc = corr_num / 10000
print 'Testing accuracy: %.4f' % test_acc
print 'P=%.4f' % P
score = P/2 + 1-test_acc
print 'Score: %.4f' % score



# -----------------------------------
# Save my precious model
# -----------------------------------
saver = tf.train.Saver()
if not os.path.exists('./model'):
    os.makedirs('./model')
saver.save(sess, './model/mnist_cnn.ckpt')
