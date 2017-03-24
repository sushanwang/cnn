#! /usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import data_helpers
import datetime as dt


class Eval():
    def __init__(self,flag):
        self.FLAGS = flag
        self.words, self.word_num_map, \
            self.vocab_processor, self.app_dict = data_helpers.load_support_dict(self.FLAGS.checkpoint_dir)
        self.x_raw, self.y_test = data_helpers.load_eval_data(self.FLAGS.data_file, self.words, self.word_num_map)
        self.y_test = np.argmax(self.y_test, axis=1)
        self.x_test = np.array(list(self.vocab_processor.transform(self.x_raw)))
        checkpoint_file = tf.train.latest_checkpoint(self.FLAGS.checkpoint_dir)
        session_conf = tf.ConfigProto(
          allow_soft_placement=self.FLAGS.allow_soft_placement,
          log_device_placement=self.FLAGS.log_device_placement)
        self.sess = tf.Session(config=session_conf)
        self.sess.as_default()
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(self.sess, checkpoint_file)

    def eval(self):
        print("\nEvaluating...\n")
        st = dt.datetime.now()
        # Get the placeholders from the graph by name
        input_x, _, app_b, dropout_keep_prob, predictions, _ = tf.get_collection('model')
        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(self.x_test), self.FLAGS.batch_size, 1, shuffle=False)
        # Collect the predictions here
        all_predictions = []
        all_test = []
        for x_test_batch in batches:
            b = data_helpers.get_W_by_x_input(x_test_batch, self.app_dict, self.word_num_map)
            batch_predictions = self.sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0, app_b: b})
            all_test.append(x_test_batch)
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        if self.y_test is not None:
            error_list = []
            correct_list = []
            res = open("dataset/cnn_res.txt","w")
            for a in range(0,len(all_predictions)):
                line = self.x_raw[a] + " | " + self.words[int(all_predictions[a])] + " | " + self.words[self.y_test[a]] + "\n"
                res.write(line)
                if self.y_test[a] != all_predictions[a]:
                    error_list.append((self.x_raw[a], self.words[int(all_predictions[a])], self.words[self.y_test[a]]))
                else:
                    correct_list.append((self.x_raw[a],self.words[self.y_test[a]]))
            res.close()
            print("************")
            print(correct_list)
            print(e for e in correct_list)
            print(len(correct_list))
            print("**************")
            print(e for e in error_list)
            print(len(error_list))
            correct_predictions = float(sum(all_predictions == self.y_test))
            print("Total number of test examples: {}".format(len(self.y_test)))
            print("Accuracy: {:g}".format(correct_predictions/float(len(self.y_test))))
            et = dt.datetime.now()
            print(et - st)
            print((et - st)/len(self.y_test))
