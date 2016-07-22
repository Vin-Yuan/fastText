from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import random
import zipfile
import time
import datetime
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from fastText import fastText
import data_helper
import ipdb


tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
vocabulary_size = 50000

batch_size = 64
embedding_size = 128  # Dimension of the embedding vector.
window_size = 6       # How many words to consider left and right.

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=False)
    sess = tf.Session(config=session_conf)
    FastText = fastText(
        vocab_size=vocabulary_size, 
        embedding_size=200,
        window_size=window_size,
        class_num=2)

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(FastText.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    # grads_and_vars is a list of tuples (gradient, variable).
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.merge_summary(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    #timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H_%M_%S")
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.scalar_summary("loss", FastText.loss)
    acc_summary = tf.scalar_summary("accuracy", FastText.accuracy)

    # Train Summaries
    train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables(), max_to_keep = 0)

    sess.run(tf.initialize_all_variables())

    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
          FastText.train_inputs: x_batch,
          FastText.train_labels: y_batch,
        }
        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op, FastText.loss, FastText.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
          FastText.train_inputs: x_batch,
          FastText.train_labels: y_batch,
        }
        step, summaries, loss, accuracy = sess.run(
            [global_step, dev_summary_op, FastText.loss, FastText.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        if writer:
            writer.add_summary(summaries, step)

    # Generate batches
    text, labels = data_helper.load_data_and_labels('./data')
    x_train, x_dev = text[:-1000], text[-1000:]
    y_train, y_dev = labels[:-1000], labels[-1000:]
    data, count, dictionary, reverse_dictionary = data_helper.build_dataset(text)
    # batches = data_helper.generate_batch(data, labels, batch_size, window_size)
    # Training loop. For each batch...
    for x_batch, y_batch in data_helper.generate_batch(data, labels, batch_size, window_size):
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        if current_step % FLAGS.evaluate_every == 0:
            print("\nEvaluation:")
            dev_step(x_dev, y_dev, writer=dev_summary_writer)
            print("")
        if current_step % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))
