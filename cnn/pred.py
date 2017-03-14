#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import learn
import data_helpers


class Pred():

    def __init__(self,flag):

        self.FLAGS = flag
        self.load_data(self.FLAGS.data_file)

    def load_data(self,file):
        # CHANGE THIS: Load data. Load your own data here
        path = os.path.join(self.FLAGS.checkpoint_dir, "..")
        self.words = data_helpers.load_obj(path,"words")
        self.all_words = data_helpers.load_obj(path,"all_words")
        self.word_num_map = data_helpers.load_obj(path,"word_num_map")
        self.FLAGS._parse_flags()
        # Map data into vocabulary
        vocab_path = os.path.join(self.FLAGS.checkpoint_dir, "..", "vocab")
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        self.app_dict = data_helpers.get_app_dict(self.vocab_processor, self.all_words)

    def restore_model(self):
        checkpoint_file = tf.train.latest_checkpoint(self.FLAGS.checkpoint_dir)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=self.FLAGS.allow_soft_placement,
              log_device_placement=self.FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            self.sess.as_default()

            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(self.sess, checkpoint_file)

            # Get the placeholders from the graph by name
            self.input_x = graph.get_operation_by_name("input_x").outputs[0]
            self.app_b = graph.get_operation_by_name("app_b").outputs[0]

            self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            self.scores = graph.get_operation_by_name("output/scores").outputs[0]
            self.W_x_b = graph.get_operation_by_name("app_b").outputs[0]

    def predict(self, x_raw):

        def get_n_max(s):
            [s] = s
            score_dict = dict(zip(range(len(s)),s))
            sorted_score = sorted(score_dict.items(), key=lambda x: -x[1])
            tmp = [(self.words[id], prob) for id, prob in sorted_score[:5]]
            return tmp

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