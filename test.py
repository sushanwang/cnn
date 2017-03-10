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
        _, _, self.words = data_helpers.load_data_and_labels(file)
        self.FLAGS._parse_flags()
        # Map data into vocabulary
        vocab_path = os.path.join(self.FLAGS.checkpoint_dir, "..", "vocab")
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)


    def pred(self,x_raw):
        x_raw = x_raw.strip()

        x = np.array(list(self.vocab_processor.transform(x_raw)))
        x_test = np.zeros_like(x[0])
        for i in range(0,len(x)):
            x_test[i] = x[i][0]
        x_test = [x_test]
        checkpoint_file = tf.train.latest_checkpoint(self.FLAGS.checkpoint_dir)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=self.FLAGS.allow_soft_placement,
              log_device_placement=self.FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                app_b = graph.get_operation_by_name("app_b").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                scores = graph.get_operation_by_name("output/scores").outputs[0]
                W_x_b = graph.get_operation_by_name("app_b").outputs[0]
                b = data_helpers.get_W_by_x_input(x_test, self.vocab_processor, self.words)
                # Collect the predictions here
                batch_predictions, score, W = sess.run([predictions, scores, W_x_b], {input_x: x_test, dropout_keep_prob: 1.0, app_b: b})
                print(W)
                print(score)
                return self.words[batch_predictions[0]]