#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import jieba
import os


def cws(input_file_path="data/test.txt", output_file_path="data/test_prob_seg.txt"):
    with codecs.open(input_file_path, "r", "utf-8") as f:
        data = [eachLine.strip() for eachLine in f.readlines()]
        data_seg = [[word for word in jieba.cut(eachLine)] for eachLine in data]

    with codecs.open(output_file_path, "w", "utf-8") as f:
        for line in data_seg:
            f.write(" ".join(line)+"\n")
    # return data, data_seg


def ner(input_file_path="data/test_prob_seg.txt", output_file_path="data/test_prob_ner.txt"):
    print("named entity recognition begin...")
    os.system("sh ner.sh {} {}".format(os.path.abspath(input_file_path), os.path.abspath(output_file_path)))
    print("named entity recognition done!")

def lex_func():
    cws()
    ner()

if __name__ == "__main__":
    if not os.path.exists(os.path.abspath("data/test_lexical.json")):
        cws()
        ner()
