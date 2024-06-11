import pickle
import time
import tensorflow as tf
import numpy as np
import random
import math
import os
import numpy as np
import json
import shelve
import csv
from scipy.sparse import csr_matrix
import torch
import pandas as pd
import matplotlib.pyplot as plt

RATINGFILE = dict({'recipe': 'pre_ratings.txt'})
LR = dict({'recipe': 5e-3})
L2 = dict({'recipe': 1e-2})
EMB = dict({'recipe': 100})
BOUND = dict({'recipe': 3.5})

def set_seed():
    seed = 42  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def pickle_save(object, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(object, f)


def pickle_load(file_path):
    f = open(file_path, 'rb')
    return pickle.load(f)

def josn_load(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def read_csv_file(file_path):
    data = []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)

        for row in reader:
            data.append(row)

    return data

def popular_in_train_user(ratingfile='recipe', topk=300, boundary=3.0):
    rating_file_path = './data/recipe/filtered_data.txt'
    rating = np.loadtxt(fname=rating_file_path, delimiter=',')

    user_set = set()
    item_set = set()
    for i, j, k in rating:
        user_set.add(int(i))
        item_set.add(int(j))

    user_num = len(user_set)
    item_num = max(list(item_set)) + 1

    data_list = []
    row_list = []
    col_list = []
    for i, j, k in rating:
        if int(k) > boundary:
            data_list.append(1.0)
            row_list.append(int(i))
            col_list.append(int(j))
        elif int(k) < boundary:
            data_list.append(-1.0)
            row_list.append(int(i))
            col_list.append(int(j))

    data = np.array(data_list)
    row = np.array(row_list)
    col = np.array(col_list)

    r_matrix = csr_matrix((data, (row, col)), shape=(user_num, item_num))

    boundary_user_id = int(user_num * 0.8)
    ave_rating = np.array(r_matrix[:boundary_user_id].mean(axis=0)).ravel()
    topk_item = np.argsort(ave_rating)[-topk:]
    print(topk_item[0])
    pickle_save(topk_item, './data/run_time/%s_pop%d.pkl' % (ratingfile, topk))

def get_envobjects(ratingfile='recipe', max_step=5000, train_rate=0.95,

                   max_stop_count=30):
    rating_file_path = './data/recipe/filtered_data.txt'
    rating = np.loadtxt(fname=rating_file_path, delimiter=',')
    lr = LR[ratingfile]
    l2_factor = L2[ratingfile]
    emb_size = EMB[ratingfile]
    boundary_rating = BOUND[ratingfile]

    user_set = set()
    item_set = set()
    for i, j, k in rating:
        user_set.add(int(i))
        item_set.add(int(j))

    user_num = len(user_set)
    item_num = len(item_set)

    data = np.array(rating)
    np.random.shuffle(data)

    t = int(len(data) * train_rate)
    dtrain = data[:t]
    dtest = data[t:]

    tf.compat.v1.disable_v2_behavior()
    max_user_id = int(np.max(dtrain[:, 0])) + 1
    max_item_id = int(np.max(dtrain[:, 1])) + 1

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    user_embeddings = tf.Variable(tf.compat.v1.truncated_normal([max_user_id, emb_size], mean=0, stddev=0.01))
    item_embeddings = tf.Variable(tf.compat.v1.truncated_normal([max_item_id, emb_size], mean=0, stddev=0.01))
    item_bias = tf.Variable(tf.zeros([max_item_id, 1], tf.float32))

    user_ids = tf.compat.v1.placeholder(tf.int32, shape=[None])
    item_ids = tf.compat.v1.placeholder(tf.int32, shape=[None])
    ys = tf.compat.v1.placeholder(tf.float32, shape=[None])

    user_ids_clipped = tf.clip_by_value(user_ids, 0, max_user_id)
    item_ids_clipped = tf.clip_by_value(item_ids, 0, max_item_id)
    ibias_ids_clipped = tf.clip_by_value(item_ids, 0, max_item_id)

    user_embs = tf.gather(user_embeddings, user_ids_clipped)
    item_embs = tf.gather(item_embeddings, item_ids_clipped)
    ibias_embs = tf.gather(item_bias, ibias_ids_clipped)
    dot_e = user_embs * item_embs

    ys_pre = tf.reduce_sum(dot_e, 1) + tf.squeeze(ibias_embs)

    target_loss = tf.reduce_mean(0.5 * tf.square(ys - ys_pre))
    loss = target_loss + l2_factor * (tf.reduce_mean(tf.square(user_embs) + tf.square(item_embs)))

    train_step = tf.compat.v1.train.AdamOptimizer(lr).minimize(loss)

    rmse = tf.sqrt(tf.reduce_mean(tf.square(ys - ys_pre)))

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        np.random.shuffle(dtrain)
        rmse_train, loss_v, target_loss_v = sess.run([rmse, loss, target_loss],
                                                     feed_dict={user_ids: dtrain[:, 0], item_ids: dtrain[:, 1],
                                                                ys: np.float32(dtrain[:, 2])})
        rmse_test = sess.run(rmse,
                             feed_dict={user_ids: dtest[:, 0], item_ids: dtest[:, 1], ys: np.float32(dtest[:, 2])})
        print('----round%2d: rmse_train: %f, rmse_test: %f, loss: %f, target_loss: %f' % (
            0, rmse_train, rmse_test, loss_v, target_loss_v))
        pre_rmse_test = 100.0
        stop_count = 0
        stop_count_flag = False
        for i in range(max_step):
            feed_dict = {user_ids: dtrain[:, 0],
                         item_ids: dtrain[:, 1],
                         ys: np.float32(dtrain[:, 2])}
            sess.run(train_step, feed_dict)
            np.random.shuffle(dtrain)
            rmse_train, loss_v, target_loss_v = sess.run([rmse, loss, target_loss],
                                                         feed_dict={user_ids: dtrain[:, 0], item_ids: dtrain[:, 1],
                                                                    ys: np.float32(dtrain[:, 2])})
            rmse_test = sess.run(rmse, feed_dict={user_ids: dtest[:, 0], item_ids: dtest[:, 1],
                                                  ys: np.float32(dtest[:, 2])})
            print('----round%2d: rmse_train: %f, rmse_test: %f, loss: %f, target_loss: %f' % (
                i + 1, rmse_train, rmse_test, loss_v, target_loss_v))
            if rmse_test > pre_rmse_test:
                stop_count += 1
                if stop_count == max_stop_count:
                    stop_count_flag = True
                    break
            pre_rmse_test = rmse_test

        user_embeddings_value, item_embeddings_value, ibias_bias_value = sess.run([user_embeddings, item_embeddings, item_bias])
        mf_rating = np.dot(user_embeddings_value, item_embeddings_value.T) + ibias_bias_value.T
        rela_num = np.sum(np.where(mf_rating > boundary_rating, 1, 0), axis=1)
        print('done with full stop count' if stop_count_flag else 'done with full training step')
        pickle_save({'user_embedding': user_embeddings_value, 'item_embedding': item_embeddings_value, 'ibias_bias': ibias_bias_value,'user_num': user_num, 'item_num': item_num, 'rela_num': rela_num},
                    './data/run_time/%s_env_objects' % ratingfile)
        return user_embeddings_value, item_embeddings_value, ibias_bias_value

def get_rating(user_id, recipe_id):
    with shelve.open('ratings_db') as db:
        key = f'{user_id}_{recipe_id}'
        rating_val = db.get(key)

    return rating_val

def plot_graph(data, y_label, title, name):
    plt.plot(data)
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.savefig(name)

if __name__ == '__main__':
    get_envobjects(ratingfile='recipe', max_step=5000, train_rate=0.95,\
                   max_stop_count=30)
    print('1')
