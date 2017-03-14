#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
import datetime as dt


class Eval():
    def __init__(self,flag):

        self.FLAGS = flag
        self.load_data(self.FLAGS.data_file)

    # currently using the same data as training, just for inspecting what errors are made during training,
    # should load new data here for real evaluation
    def load_data(self, file='dataset/query_pkg.txt'):
        path = os.path.join(self.FLAGS.checkpoint_dir, "..")
        self.words = data_helpers.load_obj(path,"words")
        self.all_words = data_helpers.load_obj(path,"all_words")
        self.word_num_map = data_helpers.load_obj(path,"word_num_map")
        self.x_raw, self.y_test, _, _, _ = data_helpers.load_data_and_labels(file)
        self.y_test = np.argmax(self.y_test, axis=1)
        vocab_path = os.path.join(self.FLAGS.checkpoint_dir, "..", "vocab")
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        self.x_test = np.array(list(self.vocab_processor.transform(self.x_raw)))
        self.app_dict = data_helpers.get_app_dict(self.vocab_processor, self.all_words)

    def eval(self):
        print("\nEvaluating...\n")

        # Evaluation
        # ==================================================
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
                st = dt.datetime.now()
                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                app_b = graph.get_operation_by_name("app_b").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                batches = data_helpers.batch_iter(list(self.x_test), self.FLAGS.batch_size, 1, shuffle=False)

                # Collect the predictions here
                all_predictions = []
                all_test = []
                for x_test_batch in batches:

                    b = data_helpers.get_W_by_x_input(x_test_batch, self.app_dict, self.word_num_map)
                    batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0, app_b: b})
                    all_test.append(x_test_batch)
                    all_predictions = np.concatenate([all_predictions, batch_predictions])

        if self.y_test is not None:

            error_list = []
            correct_list = []
            for a in range(0,len(all_predictions)):
                if self.y_test[a] != all_predictions[a]:
                    error_list.append((self.x_raw[a], self.words[int(all_predictions[a])], self.words[self.y_test[a]]))
                else:
                    correct_list.append((self.x_raw[a],self.words[self.y_test[a]]))
            print("************")
            print(correct_list)
            for e in correct_list:
                print(e)
            print(len(correct_list))

            print("**************")
            for e in error_list:
                print(e)
            print(len(error_list))

            correct_predictions = float(sum(all_predictions == self.y_test))
            print("Total number of test examples: {}".format(len(self.y_test)))
            print("Accuracy: {:g}".format(correct_predictions/float(len(self.y_test))))
            et = dt.datetime.now()
            print(et - st)
            print((et - st)/len(self.y_test))
