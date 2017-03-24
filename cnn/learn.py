#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import data_helpers
import datetime


class Learn():
    def __init__(self,flag):
        self.FLAGS = flag
        self.words, self.word_num_map, \
            self.vocab_processor, self.app_dict = data_helpers.load_support_dict(self.FLAGS.checkpoint_dir)
        session_conf = tf.ConfigProto(
          allow_soft_placement=self.FLAGS.allow_soft_placement,
          log_device_placement=self.FLAGS.log_device_placement)
        self.sess = tf.Session(config=session_conf)
        checkpoint_file = tf.train.latest_checkpoint(self.FLAGS.checkpoint_dir)
        self.sess.as_default()
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(self.sess, checkpoint_file)
        self.input_x, self.input_y, self.app_b, self.dropout_keep_prob, _, _ = tf.get_collection('model')

    def train_step(self, x_batch, y_batch, all_vars, train_summary_writer):
        """
        A single training step
        """
        app_b = data_helpers.get_W_by_x_input(x_batch, self.app_dict, self.word_num_map)
        feed_dict = {
          self.input_x: x_batch,
          self.input_y: y_batch,
          self.dropout_keep_prob: self.FLAGS.dropout_keep_prob,
          self.app_b:app_b
        }
        _, step, summaries, loss, accuracy, scores, input_y = self.sess.run(
            all_vars,
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    def learn(self, x_raw, y_raw, num=10):
        # x = [seq, vocab]
        x = np.array(list(self.vocab_processor.transform(x_raw)))
        x_test = np.zeros_like(x[0])
        for i in range(len(x)):
            x_test[i] = x[i][0]
        x_batch = [x_test] * num
        y_max_len = len(self.words)
        y_row = np.zeros([y_max_len])
        if self.word_num_map.get(y_raw):
            y_row[self.word_num_map.get(y_raw)] = 1
        y_batch = [y_row] * num
        all_vars = tf.get_collection('cnn')
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.FLAGS.num_checkpoints)
        global_step = all_vars[1]
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        train_summary_writer, checkpoint_prefix = tf.get_collection("saver")
        for i in range(self.FLAGS.num_epochs):
            current_step = tf.train.global_step(self.sess, global_step)
            self.train_step(x_batch, y_batch, all_vars, train_summary_writer)
        path = saver.save(self.sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))
        # x_test = [1, vocab]
        print("learned!")