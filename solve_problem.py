#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import codecs
import argparse

import numpy as np

from lexical import lex_func
from search_knowledge import search_know
from model_pred import nn_pred

MODEL_DIR = "data/model/"


class Model_attribute(object):
    def __init__(self, model_file, build_ac, attention_type, sum_type, loss_type, reg):
        self.model_file = model_file
        self.build_ac = build_ac
        self.attention_type = attention_type
        self.sum_type = sum_type
        self.loss_type = loss_type
        self.reg = reg


def model_ensemble(problem_number=8):
    model1_attr = Model_attribute(model_file=MODEL_DIR + 'model1.npz', build_ac=False, attention_type='dot_attention',
                                  sum_type='mlp_sum_layer', loss_type='cross_entropy', reg=0.0001)
    model2_attr = Model_attribute(model_file=MODEL_DIR + 'model2.npz', build_ac=False,
                                  attention_type='bilinear_attention', sum_type='sum_layer', loss_type='cross_entropy',
                                  reg=0.0001)
    model3_attr = Model_attribute(model_file=MODEL_DIR + 'model3.npz', build_ac=False,
                                  attention_type='bilinear_attention', sum_type='sum_layer', loss_type='max_margin',
                                  reg=0.0001)
    model4_attr = Model_attribute(model_file=MODEL_DIR + 'model4.npz', build_ac=False,
                                  attention_type='bilinear_attention', sum_type='mlp_sum_layer',
                                  loss_type='cross_entropy', reg=0.0001)
    model5_attr = Model_attribute(model_file=MODEL_DIR + 'model5.npz', build_ac=True, attention_type='dot_attention',
                                  sum_type='sum_layer', loss_type='max_margin', reg=0.0001)
    model6_attr = Model_attribute(model_file=MODEL_DIR + 'model6.npz', build_ac=True,
                                  attention_type='bilinear_attention', sum_type='sum_layer', loss_type='max_margin',
                                  reg=0.0001)

    model_attr_list = [model1_attr, model2_attr, model3_attr, model4_attr, model5_attr, model6_attr]

    result = np.zeros((len(model_attr_list), problem_number, 4))
    for i in range(len(model_attr_list)):
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
        parser.add_argument('--max_seq_len', type=int, default=300, help='Max seq len')
        parser.add_argument('--n_hidden', type=int, default=256, help='Hidden size')
        parser.add_argument('--regularizable', type=bool, default=False, help='whether to be regularizable')

        parser.add_argument('--train_data_rate', type=float, default=0.8, help='Train data rate')
        parser.add_argument('--valid_data_rate', type=float, default=0.2, help='Valid data rate')
        parser.add_argument('--choice_num', type=int, default=4, help='Choice number')
        parser.add_argument('--analysis_num', type=int, default=8, help='Analysis number')

        parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
        parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
        parser.add_argument('--grad_clip', type=int, default=30, help='Grad_clipping')
        parser.add_argument('--n_epoch', type=int, default=100, help='Number of epoch')
        parser.add_argument('--patience', type=int, default=5, help='Patience')
        parser.add_argument('--model_file', type=str, default=model_attr_list[i].model_file, help='Pre-trained model.')
        parser.add_argument('--pre_trained', type=bool, default=True, help="Whether or not pre-train")
        parser.add_argument('--shuffle', type=bool, default=True, help="Whether or not shuffle data")
        parser.add_argument('--reg', type=float, default=0.0001, help="Whether to use l2-regularization")

        parser.add_argument('--vocab_size', type=int, default=678388, help='Vocabulary size')
        parser.add_argument('--embedding_size', type=int, default=150, help='Embedding size')

        parser.add_argument('--build_ac', type=bool, default=model_attr_list[i].build_ac, help='Whether build ac')
        parser.add_argument('--attention_type', type=str, default=model_attr_list[i].attention_type,
                            help='Attention type')
        parser.add_argument('--sum_type', type=str, default=model_attr_list[i].sum_type, help='Sum type')
        parser.add_argument('--loss_type', type=str, default=model_attr_list[i].loss_type, help='Loss type')
        args = parser.parse_args()


        temp = nn_pred(args)
        result[i, :, :] = temp
    result = np.array(result)
    result = np.transpose(result, (1, 0, 2))
    result = result.mean(axis=1)

    pred = np.argmax(result, axis=1)
    logging.info("final predict value is:")
    logging.info(pred)
    logging.info("final probas is:")
    logging.info(result)

    with codecs.open("data/test_probs.txt", "w", "utf-8") as f:
        for line in result:
            temp = ["{:.6f}".format(each_value) for each_value in line]
            f.write("\t".join(temp) + "\n")
    logging.info("~" * 50)
    logging.info("problem solving done!!!")


def solve_func():
    logging.basicConfig(filename='data/test.info', filemode='w', level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')

    if not os.path.exists(os.path.abspath("data/test_lexer.json")):
        lex_func()
    if not os.path.exists(os.path.abspath("data/test_retrieval.json")):
        search_know()
    model_ensemble()


if __name__ == "__main__":
    solve_func()

