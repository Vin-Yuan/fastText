import os
import time
import numpy as np
import tensorflow as tf


class fastText(object):
    def __init__(
        self,vocab_size, embedding_size, window_size, class_num):
        # Input data.
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, window_size],name='input_x')
        self.train_labels = tf.placeholder(tf.float32, shape=[None, class_num],name='input_y')
        self.vocab_size = vocab_size
        # Ops and variables pinned to the CPU because of missing GPU implementationjjjj
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            self.embeddings = tf.Variable(
                tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0))
            # embed is shape=[None, window_size, embedding_size]
            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
        with tf.name_scope('hidden_layer'):
            Average = tf.reduce_sum(embed, reduction_indices=1) / float(embedding_size)
        # may be need l2_loss ? 
        with tf.name_scope('output'):
            W = tf.get_variable(
                "W",
                shape=[embedding_size, class_num],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[class_num]), name='b')
            self.scores = tf.nn.xw_plus_b(Average, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name='predictions')
        # CalculateMean cross-entrop loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.train_labels)
            self.loss = tf.reduce_mean(losses)
        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.train_labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')



