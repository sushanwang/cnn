# -*- coding: utf-8 -*-
import numpy as np
import collections
import pickle


def save_obj(obj, dir, name):
    with open(dir + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 2)


def load_obj(dir, name):
    with open(dir + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# Load dataset from file
def load_data_and_labels(query_file):
    query_list = []
    app_list = []
    with open(query_file, "r") as f:
        for line in f:
            try:
                query, app = line.strip().split(' | ')
                query_list.append(query)
                app_list.append(app)
            except Exception as e:
                pass
    all_words = [app for app in app_list]
    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    new_pairs = [pair for pair in count_pairs if pair[1] > 1]
    words, _ = zip(*new_pairs)
    word_num_map = dict(zip(words, range(len(words))))
    new_query_list = []
    new_app_list = []
    for query, app in zip(query_list,app_list):
        if word_num_map.get(app):
            new_query_list.append(query)
            new_app_list.append(app)
    print('命令总数: ', len(new_query_list))
    y_max_len = len(words)
    print('app classes: ',y_max_len)
    all_words = [app for app in app_list]
    app_vector = []
    for app in new_app_list:
        app_vector.append(word_num_map.get(app))
    xdata = []
    for query in new_query_list:
        s = ""
        for word in query:
            s += word + " "
        xdata.append(s.strip())
    ydata = []
    for app in app_vector:
        y_row = np.zeros([y_max_len])
        y_row[app] = 1
        ydata.append(y_row)
    print("data done!")
    return xdata, np.asarray(ydata), words, all_words, word_num_map


def load_eval_data(query_file, words, word_num_map):
    query_list = []
    app_list = []
    print(query_file)
    with open(query_file, "r") as f:
        for line in f:
            try:
                query, app = line.strip().split(' | ')
                query_list.append(query)
                app_list.append(app)

            except Exception as e:
                pass
    y_max_len = len(words)
    xdata = []
    ydata = []
    for query, app in zip(query_list,app_list):
        if word_num_map.get(app):
            s = ""
            for word in query:
                s += word + " "
            xdata.append(s.strip())
            y_row = np.zeros([y_max_len])
            y_row[word_num_map.get(app)] = 1
            ydata.append(y_row)
    print("data done!")
    return xdata, np.asarray(ydata)


# transform name string into name vector
def transform_app_name_to_vector(vocab_processor, a):
    t = [np.array(list(vocab_processor.fit_transform(a)))]
    for a in t:
        new_line = [line[0] for line in a]
    return new_line


def get_app_dict(vocab_processor, all_words):

    app_dict = {}
    with open("dataset/app_pkg.txt") as f:
        lines = f.readlines()
        for line in lines:
            app, pkg = line.split(' | ')
            if all_words.count(pkg.strip()) != 0:
                app_vec = transform_app_name_to_vector(vocab_processor, app)
                if app_dict.get(app_vec[0]):
                    app_dict[app_vec[0]].append((app_vec[1:], pkg.strip()))
                else:
                    app_dict[app_vec[0]] = [(app_vec[1:], pkg.strip())]
    return app_dict


# construct app_W matrix by searching app name in query
def get_W_by_x_input(x_input, app_dict, word_num_map):
    W = np.zeros([len(x_input), len(word_num_map.values())], dtype=np.float32)
    new_x = []
    for x in x_input:
        t = []
        for a in x:
            if a != 0:
                t.append(a)
        new_x.append(t)
    j = 0
    for query in new_x:
        added_app = []
        for i in range(len(query)):
            if app_dict.get(query[i]):
                l = app_dict.get(query[i])
                for vec, name in l:
                    if vec == query[i+1:i+1+len(vec)]:
                        W[j][word_num_map.get(name.strip())] = 1
                        added_app.append((vec, name))
        for k in range(len(added_app)):
            for ii in range(len(added_app)):
                if len(set(added_app[ii][0]).difference(added_app[k][0])) == 0 and k != ii:
                    W[j][word_num_map.get(added_app[ii][1])] = 0
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
