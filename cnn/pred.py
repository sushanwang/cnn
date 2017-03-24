#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import data_helpers


class Pred():
    def __init__(self,flag):
        self.FLAGS = flag
        self.words, self.word_num_map, \
            self.vocab_processor, self.app_dict = data_helpers.load_support_dict(self.FLAGS.checkpoint_dir)
        checkpoint_file = tf.train.latest_checkpoint(self.FLAGS.checkpoint_dir)
        session_conf = tf.ConfigProto(
          allow_soft_placement=self.FLAGS.allow_soft_placement,
          log_device_placement=self.FLAGS.log_device_placement)
        self.sess = tf.Session(config=session_conf)
        self.sess.as_default()
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(self.sess, checkpoint_file)
        self.input_x, _, self.app_b, self.dropout_keep_prob,\
            self.predictions, self.scores = tf.get_collection('model')

    def predict(self, x_raw):
        def get_n_max(s):
            [s] = s
            score_dict = dict(zip(range(len(s)),s))
            sorted_score = sorted(score_dict.items(), key=lambda x: -x[1])
            tmp = [(self.words[id], prob) for id, prob in sorted_score[:5]]
            return tmp
        x_raw = x_raw.replace(" ", "")
        x_raw = x_raw.strip()
        # x = [seq, vocab]
        x = np.array(list(self.vocab_processor.transform(x_raw)))
        x_test = np.zeros_like(x[0])
        for i in range(len(x)):
            x_test[i] = x[i][0]
        x_test = [x_test]
        # x_test = [1, vocab]
        b = data_helpers.get_W_by_x_input(x_test, self.app_dict, self.word_num_map)
        # Collect the predictions here
        batch_predictions, score = self.sess.run([self.predictions, self.scores], {self.input_x: x_test, self.dropout_keep_prob: 1.0, self.app_b: b})
        return get_n_max(score)