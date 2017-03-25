# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy import arange

import argparse
import sys
import numpy as np
import dataParser as dataParser
import tensorflow as tf
import neuralNet as nn

FLAGS = None

def main(_):
  # Define loss and optimizer
  (classes, ingredients, X, y, y_cuisine, all_classes) = dataParser.parse_input('train.json')
  # dataset = dataParser.parse_input('train.json')
  with tf.Graph().as_default():
    # Predictive model (Neural Net)
    # inputX = tf.placeholder(tf.float32, [None, nn.FEATURES])
    inputX = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, nn.FEATURES))
    # outputY = tf.placeholder(tf.float32, [None, FLAGS.multi_Classes])
    outputY = tf.placeholder(tf.int32, shape=(FLAGS.batch_size, FLAGS.multi_Classes))

    # Predictive model
    y_hat = nn.y_hat(inputX, FLAGS.hidden1, FLAGS.hidden2, FLAGS.hidden3)

    # Cost and optimization
    loss = nn.loss(y_hat, outputY)  
    train_step = nn.training(loss, FLAGS.learning_rate)

    # labels = tf.cast(X, tf.float32)
    # print(X.shape[0])

    # Session
    sess = tf.InteractiveSession() 
    tf.global_variables_initializer().run()

    # Train
    for _ in range(1, 100):
      indices = np.random.choice(X.shape[0], 50)[0]
      x_cast = tf.cast(X, tf.float32)
      y_cuisine_cast = tf.cast(y_cuisine, tf.float32)
      (batch_xs, batch_ys) = x_cast[indices:], y_cuisine_cast[indices:]
      batch_xs_cast = tf.cast(batch_xs, tf.float32)
      batch_ys_cast = tf.cast(batch_ys, tf.float32)
      if len < 0:
        sess.run(train_step, feed_dict={np.ndarray(X): np.ndarray(batch_xs_cast), np.ndarray(y_cuisine): np.ndarray(batch_ys_cast)})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(outputY, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={inputX: X,
                                        outputY: y_cuisine}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--hidden1',
      type=int,
      default=256,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=128,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--hidden3',
      type=int,
      default=128,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--learning_rate',
      type=int,
      default=0.4,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=1000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--multi_Classes',
      type=int,
      default=20,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=39774,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--data_size',
      type=int,
      default=1000,
      help='data size needs help.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  run_training()

