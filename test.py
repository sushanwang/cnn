#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib import learn
import data_helpers
import datetime as dt

class Pred():

    def __init__(self,flag):

        self.FLAGS = flag
        self.load_data(self.FLAGS.data_file)

    def load_data(self,file):
        # CHANGE THIS: Load data. Load your own data here
        _, _, self.words = data_helpers.load_data_and_labels(file)
        self.FLAGS._parse_flags()
        # Map data into vocabulary
        vocab_path = os.path.join(self.FLAGS.checkpoint_dir, "..", "vocab")
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

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
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            # Tensors we want to evaluate
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            self.scores = graph.get_operation_by_name("output/scores").outputs[0]
            self.W_x_b = graph.get_operation_by_name("app_b").outputs[0]

    def pred(self, x_raw):
        """
        class ScoreList():
            def __init__(self):
                self.score = None
                self.prob = 0.0
        """

        def get_N_max(score):
            [score] = score
            score_dict = dict(zip(range(len(score)),score))
            sorted_score = sorted(score_dict.items(), key=lambda x: -x[1])
            tmp = [(self.words[id], prob) for id, prob in sorted_score[:5]]
            return tmp
        x_raw = x_raw.strip()
        x = np.array(list(self.vocab_processor.transform(x_raw)))
        x_test = np.zeros_like(x[0])
        for i in range(len(x)):
            x_test[i] = x[i][0]
        x_test = [x_test]

        b = data_helpers.get_W_by_x_input(x_test, self.vocab_processor, self.words)
        # Collect the predictions here
        batch_predictions, score = self.sess.run([self.predictions, self.scores], {self.input_x: x_test, self.dropout_keep_prob: 1.0, self.app_b: b})
        return get_N_max(score)