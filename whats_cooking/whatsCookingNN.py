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

import argparse
import sys
import numpy as np
import dataParser as dataParser
import tensorflow as tf
import neuralNet as nn

FLAGS = None

def main(_):
  # Define loss and optimizer
  classes, ingredients, X, y = dataParser.parse_input("train.json")
  trainDataset = X, y
  with tf.Graph().as_default():
    # Predictive model (Neural Net)
    inputX = tf.Variable(X, tf.float32)
    y_hat = nn.y_hat(inputX, FLAGS.hidden1, FLAGS.hidden2, FLAGS.hidden3)
    # Cost and optimization
    loss = nn.loss(y_hat, y)  
    train_step = nn.training(loss, FLAGS.learning_rate)
    # Session
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(FLAGS.max_steps):
      indices = np.random.choice(data_size, batch_size)
      batch_xs, batch_ys = X_data[indices], y_data[indices]
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: X,
                                        y_: y}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
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
      type=float,
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
      '--batch_size',
      type=int,
      default=100,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--data_size',
      type=int,
      default=1000,
      help='data size needs help.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  run_training()

