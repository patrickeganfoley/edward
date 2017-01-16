#!/usr/bin/env python
"""Mixture density network (Bishop, 1994).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Categorical, Mixture, Normal
from tensorflow.contrib import slim
from sklearn.model_selection import train_test_split


def build_toy_dataset(N):
  y_data = np.random.uniform(-10.5, 10.5, N).astype(np.float32)
  r_data = np.random.normal(size=N).astype(np.float32)  # random noise
  x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
  x_data = x_data.reshape((N, 1))
  return train_test_split(x_data, y_data, random_state=42)


def neural_network(X):
  """mu, sigma, logits = NN(x; theta)"""
  hidden1 = slim.fully_connected(X, 25)
  hidden2 = slim.fully_connected(hidden1, 25)
  mus = slim.fully_connected(hidden2, K, activation_fn=None)
  sigmas = slim.fully_connected(hidden2, K, activation_fn=tf.exp)
  logits = slim.fully_connected(hidden2, K, activation_fn=None)
  return mus, sigmas, logits


ed.set_seed(42)

N = 6000  # number of data points
N_train = 4500  # number of training data points
D = 1  # number of features
K = 10  # number of mixture components

# DATA
X_train, X_test, y_train, y_test = build_toy_dataset(N)
print("Size of features in training data: {:s}".format(X_train.shape))
print("Size of output in training data: {:s}".format(y_train.shape))
print("Size of features in test data: {:s}".format(X_test.shape))
print("Size of output in test data: {:s}".format(y_test.shape))

# MODEL
X_ph = tf.placeholder(tf.float32, [None, D])
y_ph = tf.placeholder(tf.float32, [None])

mus, sigmas, logits = neural_network(X_ph)
# TODO
# Create mixture random variable of unspecified shape; doesn't work
# due to its sample_n.
# cat = Categorical(logits=tf.fill([tf.shape(X_ph)[0], 10], 0.0))
cat = Categorical(logits=tf.zeros([N_train, K]))
components = [Normal(mu=mu, sigma=sigma) for mu, sigma
              in zip(tf.unstack(tf.transpose(mus)),
                     tf.unstack(tf.transpose(sigmas)))]
y = Mixture(cat=cat, components=components)

# INFERENCE
# There are no latent variables to infer. Thus inference is concerned
# with only training model parameters, which are baked into how we
# specify the neural networks.
inference = ed.MAP(data={y: y_ph})
inference.initialize(logdir='log')

init = tf.global_variables_initializer()
init.run()

n_epoch = 20
train_loss = np.zeros(n_epoch)
test_loss = np.zeros(n_epoch)
for i in range(n_epoch):
  info_dict = inference.update(feed_dict={X_ph: X_train, y_ph: y_train})
  train_loss[i] = info_dict['loss']
  test_loss[i] = 0.0
  # test_loss[i] = sess.run(inference.loss, feed_dict={X_ph: X_test, y_ph: y_test})
  print("Train Loss: {:0.3f}, Test Loss: {:0.3f}".format(
      train_loss[i], test_loss[i]))

# sess = ed.get_session()
# pred_weights, pred_means, pred_std = sess.run(
#     [mus, sigmas, logits], feed_dict={X_ph: X_test})
