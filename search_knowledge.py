#!/usr/bin/env python
# -*- coding: utf-8 -*-


import codecs
import logging
import os

import jieba

from gensim import corpora, models, similarities

def cws(filename):
    with codecs.open(filename, "r", "utf-8") as f:
        data = [eachLine.strip() for eachLine in f.readlines()]
        data_seg = [[word for word in jieba.cut(eachLine)] for eachLine in data]

    return data, data_seg


def search_know(problem_filename="data/test.txt", knowledge_filename="data/gold/knowledge.txt",
                                   output_filename="data/test_ana_score.txt", ana_num=8):
    with codecs.open(problem_filename, "r", "utf-8") as f:
        problem_raw = f.readlines()

    if len(problem_raw) % 6 != 0:
        logging.error("please submit a valid file! The length of the problems is {}，not a multiple of 6".format(len(problem_raw)))
        os._exit(1)

    logging.info("search related knowledge for {}".format(problem_filename))
    problem, problem_seg = cws(problem_filename)
    knowledge, knowledge_seg = cws(knowledge_filename)

    question_background_seg = [problem_seg[i] + problem_seg[i + 1] for i in range(0, len(problem_seg), 6)]

    # 转化成一组向量，向量中的元素是一个二元组（编号、频次数），对应分词后的文档中的每一个词。
    dictionary = corpora.Dictionary(knowledge_seg)
    knowledge_bow = [dictionary.doc2bow(doc) for doc in knowledge_seg]
    tfidf = models.TfidfModel(knowledge_bow)
    index = similarities.SparseMatrixSimilarity(tfidf[knowledge_bow], num_features=len(dictionary.keys()))

    question_background_bow = [dictionary.doc2bow(doc) for doc in question_background_seg]
    output = []
    problem_number = len(problem_seg) // 6
    for ques_id in range(problem_number):

        output.extend(problem_seg[6 * ques_id: 6 * (ques_id + 1)])

        sim = index[tfidf[question_background_bow[ques_id]]]
        sim_sorted = sorted(enumerate(sim), key=lambda item: -item[1])[0:ana_num]

        knowledge_related = []
        for doc_id, score in sim_sorted:  # sim_sorted = [(doc_id, score), ... (doc_id, score)]
            line = [word for word in knowledge_seg[doc_id]]
            line.append("{:.6f}".format(score))
            knowledge_related.append(line)

        output.extend(knowledge_related)

    logging.info("len(ques and ana): {}".format(len(output)))
    with codecs.open(output_filename, "w", "utf-8") as f:
        for line in output:
            f.write(" ".join(line) + "\n")

    split_problem_knowledge()
    logging.info("named entity recognition begin......")
    os.system("sh ner.sh /home/louc/graduation_design/data/test_prob_seg.txt /home/louc/graduation_design/data/test_prob_ner.txt")
    logging.info("named entity recognition done!")
    return problem_number


def remove_score(sentence, ana_num=8):
    for i in range(0, len(sentence), 6 + ana_num):
        for j in range(ana_num):
            line = [word for word in sentence[i + 6 + j].strip().split(" ")]
            line = " ".join(line[:-1]) + "\n"
            sentence[i + 6 + j] = line
    return sentence


def split_problem_knowledge(filename="data/test_ana_score.txt", ana_num=8):
    logging.info("split problem and ana......")
    with codecs.open(filename, "r", "utf-8") as f:
        data = f.readlines()

    data = remove_score(data)

    problem, ana = [], []
    for i in range(0, len(data)):
        remainder = i % (6 + ana_num)
        if remainder >= 0 and remainder < 6:
            problem.append(data[i])
        elif remainder >= 6 and remainder < 6 + ana_num:
            ana.append(data[i])

    logging.info("problem len: {}, ana len: {}".format(len(problem), len(ana)))
    with codecs.open("data/test_prob_seg.txt", "w", "utf-8") as f:
        f.writelines(problem)

    with codecs.open("data/test_ana_seg.txt", "w", "utf-8") as f:
        f.writelines(ana)