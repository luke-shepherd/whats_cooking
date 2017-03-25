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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

NUM_CLASSES = 20

FEATURES = 6714

def y_hat(inputX, hidden1_units, hidden2_units, hidden3_units):
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([FEATURES, hidden1_units], stddev=1.0 / math.sqrt(float(FEATURES))),name='weights')

    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(inputX, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Hidden 3
  with tf.name_scope('hidden3'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, hidden3_units],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden3_units]),
                         name='biases')
    hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden3_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden3_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden3, weights) + biases
  return logits

# loss = nn.loss(y_hat, outputY)  
def loss(logits, labels):  
  labels = tf.cast(labels, tf.float32)
  cross_entropy = -tf.reduce_sum(logits*tf.log(labels), name='xentropy')
  return cross_entropy


def training(loss, learning_rate):
  tf.summary.scalar('loss', loss)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


# def evaluation(logits, labels):
#   correct = tf.nn.in_top_k(logits, labels, 1)
#   return tf.reduce_sum(tf.cast(correct, tf.int32))
