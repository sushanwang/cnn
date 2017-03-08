# -*- coding: utf-8 -*-
import numpy as np
import re
import itertools
import collections


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(query_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    print(y)
    return [x_text, y]
    """


    query_list = []
    app_list = []
    with open(query_file, "r") as f:
        for line in f:
            try:
                query,app = line.strip().split(' | ')
                query_list.append(query)
                app_list.append(app)

            except Exception as e:
                pass

    print('命令总数: ', len(query_list))
    app_ids = list(set(app_list))
    y_max_len = len(app_ids)
    print('app classes: ',y_max_len)

    all_words = [app for app in app_list]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    words, _ = zip(*count_pairs)
    word_num_map = dict(zip(words, range(len(words))))
    vocab_size = len(words)
    print('apps size:', (vocab_size))

    app_vector = []
    for app in app_list:
        app_vector.append(word_num_map.get(app))
    xdata = []
    for query in query_list:
        str = ""
        for word in query:
            str += word +" "
        xdata.append(str.strip())
    ydata = []
    id = 0
    for app in app_vector:
        y_row = np.zeros([y_max_len])
        y_row[app] = 1
        ydata.append(y_row)
        id +=1
    print("data done!")
    return xdata,np.asarray(ydata),words


def get_W_by_x_input(x_input, vocab_processor, words):

    W = np.zeros([len(x_input), len(words)], dtype = np.float32)
    docs = vocab_processor.reverse(x_input)
    new_docs = []
    for raw in docs:
        line = ""
        for word in raw:
            if word != " ":
                line += word
        new_docs.append(line)
    j = 0
    for raw in new_docs:
        for i in range(len(words)):
            app_id = raw.count(words[i])
            if app_id != 0:
                W[j][i] = app_id
        j += 1
    return W







def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
