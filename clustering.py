#!/usr/bin/env python
# -*- coding:UTF-8-*-

import os
import codecs
import cPickle

from sklearn.cluster import KMeans
import gensim
import jieba
import theano
import numpy as np


def change_limit(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] == "/":
                data[i][j] = "|"
    return data


def know_ner():
    with codecs.open("data/gold/knowledge.txt", "r", "utf-8") as f:
        data = [eachLine.strip() for eachLine in f.readlines()]
        data_seg = [[word for word in jieba.cut(eachLine)] for eachLine in data]
        data_seg = change_limit(data_seg)

    with codecs.open("data/gold/knowledge_seg.txt", "w", "utf-8") as f:
        for line in data_seg:
            f.writelines(" ".join(line)+"\n")

    os.system("sh ner.sh /home/louc/graduation_design/data/knowledge_seg.txt /home/louc/graduation_design/data/knowledge_ner.txt")


def gen_knowledge_mean_vector():
    '''
    由ner和word2vec生成knowledge_vector
    :return:
    '''
    with codecs.open("data/gold/knowledge_ner.txt", "r", "utf-8") as f:
        data_ner = []
        all_line = [eachLine.strip().split(" ") for eachLine in f.readlines()]
        for line in all_line:
            temp = []
            for word in line:
                ner_label = word.split("/")[1]
                if ner_label == "OTHER":
                    temp.append([1, 0, 0])
                elif ner_label == "TIME":
                    temp.append([0, 1, 0])
                elif ner_label == "LOC":
                    temp.append([0, 0, 1])
            data_ner.append(temp)

    print("data ner one hot done!")

    word2index_map = cPickle.load(open('data/gold/word2index.pkl', 'r'))
    index2vec = np.load('data/gold/index2vec.npy')

    with codecs.open("data/gold/knowledge.txt", "r", "utf-8") as f:
        data_seg = [[word for word in eachLine.strip().split(" ")] for eachLine in f.readlines()]

    knowledge_line = []
    for i in range(len(data_seg)):
        line = []
        for j in range(len(data_seg[i])):
            if data_seg[i][j] in word2index_map:
                line.append(index2vec[word2index_map[data_seg[i][j]]].tolist())
            else:
                line.append(index2vec[0].tolist())
        knowledge_line.append(line)

    print("data text2vec done!")

    knowledge_matrix = []
    for i in range(len(knowledge_line)):
        line = [0 for _ in range(153)]
        for j in range(len(knowledge_line[i])):
            temp = knowledge_line[i][j]+data_ner[i][j]
            if len(line) == 0:
                line = [k for k in temp]
            else:
                line = [line[k]+temp[k] for k in range(153)]
        print(line, len(knowledge_line[i]))
        line = [line[k]//(len(knowledge_line[i]) * 1.0) for k in range(len(line))]
        knowledge_matrix.append(line)

    knowledge_matrix = np.array(knowledge_matrix).astype(theano.config.floatX)
    print(knowledge_matrix.shape)
    print(knowledge_matrix[10])
    np.save("data/gold/knowledge_matrix.npy", knowledge_matrix)

def cluster_kmeans(cluster_size):
    knowledge_matrix = np.load("data/gold/knowledge_matrix.npy")
    print(knowledge_matrix[11])
    print("cluster begin!")
    km = KMeans(n_clusters=cluster_size, random_state=41).fit(knowledge_matrix)
    np.save("data/gold/km.npy", km.cluster_centers_)
    print("cluster done!")

if __name__ == "__main__":
    gen_knowledge_mean_vector()
    cluster_kmeans(1000)
