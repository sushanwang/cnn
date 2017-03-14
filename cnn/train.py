#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn


class Train():
    def __init__(self,flag):
        self.FLAGS = flag
        self.load_data(self.FLAGS.data_file)

    def load_data(self, file='dataset/query_pkg.txt'):
        # Load data
        print("Loading data...")
        x_text, y, self.words, self.all_words, self.word_num_map = data_helpers.load_data_and_labels(file)
        # Build vocabulary
        max_document_length = max([len(x) for x in x_text])
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(self.vocab_processor.fit_transform(x_text)))
        self.app_dict = data_helpers.get_app_dict(self.vocab_processor, self.all_words)
        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        # Split train/test set
        dev_sample_index = -1 * int(self.FLAGS.dev_sample_percentage * float(len(y)))
        self.x_train, self.x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        self.y_train, self.y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        print("Vocabulary Size: {:d}".format(len(self.vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(self.y_train), len(self.y_dev)))

    def train(self):
        # Training
        # ==================================================
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=self.FLAGS.allow_soft_placement,
              log_device_placement=self.FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                cnn = TextCNN(
                    sequence_length=self.x_train.shape[1],
                    num_classes=self.y_train.shape[1],
                    vocab_size=len(self.vocab_processor.vocabulary_),
                    embedding_size=self.FLAGS.embedding_dim,
                    filter_sizes=list(map(int, self.FLAGS.filter_sizes.split(","))),
                    num_filters=self.FLAGS.num_filters,
                    l2_reg_lambda=self.FLAGS.l2_reg_lambda)
                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)
                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))
                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", cnn.loss)
                acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.FLAGS.num_checkpoints)
                # Write vocabulary
                self.vocab_processor.save(os.path.join(out_dir, "vocab"))
                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                def train_step(x_batch, y_batch):
                    """
                    A single training step
                    """
                    app_b = data_helpers.get_W_by_x_input(x_batch, self.app_dict, self.word_num_map)
                    feed_dict = {
                      cnn.input_x: x_batch,
                      cnn.input_y: y_batch,
                      cnn.dropout_keep_prob: self.FLAGS.dropout_keep_prob,
                      cnn.app_b:app_b
                    }
                    _, step, summaries, loss, accuracy, scores, input_y = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.scores,cnn.input_y],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()

                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

                def dev_step(x_batch, y_batch, writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    app_b = data_helpers.get_W_by_x_input(x_batch, self.app_dict, self.word_num_map)
                    feed_dict = {
                      cnn.input_x: x_batch,
                      cnn.input_y: y_batch,
                      cnn.dropout_keep_prob: 1.0,
                      cnn.app_b:app_b
                    }
                    step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    if writer:
                        writer.add_summary(summaries, step)

                # Generate batches
                batches = data_helpers.batch_iter(
                    list(zip(self.x_train, self.y_train)), self.FLAGS.batch_size, self.FLAGS.num_epochs)

                # Training loop. For each batch...
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % self.FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_step(self.x_dev, self.y_dev, writer=dev_summary_writer)
                        print("")
                    if current_step % self.FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
