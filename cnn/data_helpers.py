# -*- coding: utf-8 -*-
import numpy as np
import collections


# Load dataset from file
def load_data_and_labels(query_file):
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


# construct app_W matrix by searching app name in query
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
            if len(words[i]) > 2:
                for k in range(len(words[i])):
                    app_id = raw.count(words[i][k:2+k])
                    if app_id != 0:
                        W[j][i] = app_id
            else:
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
