#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import cPickle
import codecs
import logging
import timeit
import warnings
import gzip
import os

import lasagne
import numpy as np
import theano
import theano.tensor as T

from gensim import corpora, models, similarities
import jieba
import codecs

warnings.filterwarnings('ignore', '.*topo.*')

MODEL_DIR = 'data/model/'
ROOT_DIR = 'data/'

VECTOR_FILE = ROOT_DIR + 'gold/word2vec.txt'
WORD2INDEX_PKL_FILE = ROOT_DIR + 'gold/word2index.pkl'
INDEX2VEC_NPY_FILE = ROOT_DIR + 'gold/index2vec.npy'
UNK = u'$$$_UNK_$$$'


class EmbeddingUnregularizeLayer(lasagne.layers.Layer):
    '''
    for word2vec
    '''
    def __init__(self, incoming, input_size, output_size,
                 W=lasagne.init.Normal(), **kwargs):
        super(EmbeddingUnregularizeLayer, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size = output_size

        self.W = self.add_param(W, (input_size, output_size), name="W", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size,)

    def get_output_for(self, input, **kwargs):
        return self.W[input]


class SumLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, num_units, **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        super(SumLayer, self).__init__(incomings, **kwargs)
        self.num_units = num_units

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        return inputs[0] + inputs[1]


class KMEmbeddingLayer(lasagne.layers.Layer):
    '''
    for knowledge matrix
    '''
    def __init__(self, incoming, input_size, output_size,
                 W=lasagne.init.Normal(), **kwargs):
        super(KMEmbeddingLayer, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size = output_size

        self.W = self.add_param(W, (input_size, output_size), name="W", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size,)

    def get_output_for(self, input, **kwargs):
        return self.W[input]


class MLPSumLayer(lasagne.layers.MergeLayer):
    """
        analysis and ques mlp sum layer
        incomings[0]: analysis, shape = (batch_size, hidden_size)
        incomings[1]: question, shape = (batch_size, hidden_size)
    """

    def __init__(self, incomings, num_units, W_init=lasagne.init.GlorotUniform(), b_init=lasagne.init.Constant(0.),
                 **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        super(MLPSumLayer, self).__init__(incomings, **kwargs)
        self.num_units = num_units
        self.W1 = self.add_param(W_init, (self.num_units, self.num_units), name='W_ques')
        self.W2 = self.add_param(W_init, (self.num_units, self.num_units), name='W_know')
        self.b = self.add_param(b_init, (self.num_units,), name='b')

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, **kwargs):
        return T.dot(inputs[0], self.W1) + T.dot(inputs[1], self.W2) + self.b.dimshuffle('x', 0)


class BatchedDotLayer(lasagne.layers.MergeLayer):
    '''
	score pairs (question and 4 choices)
	incomings[0]: choices, shape=(batch_size, 4, hidden_size)
	incomings[1]: ques, shape=(batch_size, hidden_size)
	'''

    def __init__(self, incomings, **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        super(BatchedDotLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1][0], input_shapes[1][2]

    def get_output_for(self, inputs, **kwargs):
        return T.batched_dot(inputs[0], inputs[1])


class DotAttentionLayer(lasagne.layers.MergeLayer):
    '''
    question and ana attention layer
    incomings[0]: analysis, shape = (batch_size, 10, hidden_size)
    incomings[1]: question, shape = (batch_size, hidden_size)
    '''

    def __init__(self, incomings, num_units, **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError
        super(DotAttentionLayer, self).__init__(incomings, **kwargs)
        self.num_units = num_units

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):
        alpha = T.nnet.softmax(T.batched_dot(inputs[0], inputs[1]))
        alpha = T.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))
        return T.sum(inputs[0] * alpha, axis=1)


class BilinearAttentionLayer(lasagne.layers.MergeLayer):
    """
		question and ana attention layer
		incomings[0]: analysis, shape = (batch_size, 10, hidden_size)
		incomings[1]: question, shape = (batch_size, hidden_size)
	"""

    def __init__(self, incomings, num_units, init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 2:
            raise NotImplementedError

        super(BilinearAttentionLayer, self).__init__(incomings, **kwargs)
        self.num_units = num_units
        self.W = self.add_param(init, (self.num_units, self.num_units), name='W_bilinear')

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):
        # inputs[0]: batch * len * h
        # inputs[1]: batch * h
        # W: h * h

        M = T.dot(inputs[1], self.W).dimshuffle(0, 'x', 1)
        alpha = T.nnet.softmax(T.sum(inputs[0] * M, axis=2))
        return T.sum(inputs[0] * alpha.dimshuffle(0, 1, 'x'), axis=1)


class DotAttentionLayer_a(lasagne.layers.MergeLayer):
    '''
    question and ana attention layer
    incomings[0]: analysis, shape = (batch_size, 10, hidden_size)
    incomings[1]: question, shape = (batch_size, hidden_size)
    incomings[2]: analysis, shape = (batch_size, 10, hidden_size) for attention
    '''

    def __init__(self, incomings, **kwargs):
        if len(incomings) != 3:
            raise NotImplementedError
        super(DotAttentionLayer_a, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):
        alpha = T.nnet.softmax(T.batched_dot(inputs[2], inputs[1]))
        alpha = T.reshape(alpha, (alpha.shape[0], alpha.shape[1], 1))
        return T.sum(inputs[0] * alpha, axis=1)


class BilinearAttentionLayer_a(lasagne.layers.MergeLayer):
    """
		question and ana attention layer
		incomings[0]: analysis, shape = (batch_size, 10, hidden_size)
		incomings[1]: question, shape = (batch_size, hidden_size)
	"""

    def __init__(self, incomings, num_units, init=lasagne.init.Uniform(), **kwargs):
        if len(incomings) != 3:
            raise NotImplementedError

        super(BilinearAttentionLayer_a, self).__init__(incomings, **kwargs)
        self.num_units = num_units
        self.W = self.add_param(init, (self.num_units, self.num_units), name='W_bilinear')

    def get_output_shape_for(self, input_shapes):
        return input_shapes[1]

    def get_output_for(self, inputs, **kwargs):
        # inputs[0]: batch * len * h
        # inputs[1]: batch * h
        # W: h * h

        M = T.dot(inputs[1], self.W).dimshuffle(0, 'x', 1)
        alpha = T.nnet.softmax(T.sum(inputs[2] * M, axis=2))
        return T.sum(inputs[0] * alpha.dimshuffle(0, 1, 'x'), axis=1)


class Model(object):

    def __init__(self, batch_size, max_seq_len, n_hidden, regularizable, train_data_rate, valid_data_rate, choice_num,
                 analysis_num, learning_rate,
                 dropout_rate, grad_clip, n_epoch, patience, model_file, pre_trained, shuffle, reg, vocab_size,
                 embedding_size,
                 build_ac, attention_type, sum_type, loss_type):

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_hidden = n_hidden
        self.regularizable = regularizable

        self.train_data_rate = train_data_rate
        self.valid_data_rate = valid_data_rate
        self.use_valid_data = False
        if self.valid_data_rate > 0:
            self.use_valid_data = True

        self.choice_num = choice_num
        self.analysis_num = analysis_num
        self.use_analysis = False
        if self.analysis_num > 0:
            self.use_analysis = True

        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.dropout_rate = dropout_rate
        self.n_epoch = n_epoch
        self.patience = patience
        self.model_file = model_file
        self.pre_trained = pre_trained
        self.shuffle = shuffle
        self.reg = reg

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.build_ac = build_ac
        self.attention_type = attention_type
        self.sum_type = sum_type
        self.loss_type = loss_type

        self.cluster_size = 1000

        # self.make_vec_dict_file()
        self.load_word2index()
        self.load_index2vec()
        self.load_km()


    def build_model_use_analysis(self):
        '''
        build model: knowledge retrieval and knowledge encoder
        '''
        logging.info("building network ...")
        Ques = T.matrix('Ques', dtype='int64')
        Ques_mask = T.matrix('Ques_mask', dtype=theano.config.floatX)
        Choice = T.tensor3('Choice', dtype='int64')
        Choice_mask = T.tensor3('Choice_mask', dtype=theano.config.floatX)
        Analysis = T.tensor3('Analysis', dtype='int64')
        Analysis_mask = T.tensor3('Analysis_mask', dtype=theano.config.floatX)

        EmbeddingLayer = EmbeddingUnregularizeLayer
        if self.regularizable:
            EmbeddingLayer = lasagne.layers.EmbeddingLayer

        ques_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.max_seq_len))
        ques_in_embed = EmbeddingLayer(ques_in, input_size=self.vocab_size, output_size=self.embedding_size,
                                       W=self.index2vec)
        # dropout layer
        ques_in_embed = lasagne.layers.DropoutLayer(ques_in_embed, p=self.dropout_rate)

        ques_mask = lasagne.layers.InputLayer(shape=(self.batch_size, self.max_seq_len))
        ques_forward = lasagne.layers.GRULayer(ques_in_embed, self.n_hidden, mask_input=ques_mask,
                                               grad_clipping=self.grad_clip)
        ques_backward = lasagne.layers.GRULayer(ques_in_embed, self.n_hidden, mask_input=ques_mask,
                                                grad_clipping=self.grad_clip,
                                                backwards=True)
        ques_forward_slice = lasagne.layers.SliceLayer(ques_forward, -1, 1)
        ques_backward_slice = lasagne.layers.SliceLayer(ques_backward, 0, 1)
        ques_concat = lasagne.layers.ConcatLayer([ques_forward_slice, ques_backward_slice], axis=-1)

        choice_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.choice_num, self.max_seq_len))
        choice_in_reshape = lasagne.layers.ReshapeLayer(choice_in,
                                                        shape=(self.batch_size * self.choice_num, self.max_seq_len))
        choice_in_embed = EmbeddingLayer(choice_in_reshape, input_size=self.vocab_size, output_size=self.embedding_size,
                                         W=self.index2vec)
        # dropout layer
        choice_in_embed = lasagne.layers.DropoutLayer(choice_in_embed, p=self.dropout_rate)

        choice_mask = lasagne.layers.InputLayer(shape=(self.batch_size, self.choice_num, self.max_seq_len))
        choice_mask_reshape = lasagne.layers.ReshapeLayer(choice_mask,
                                                          shape=(self.batch_size * self.choice_num, self.max_seq_len))
        choice_forward = lasagne.layers.GRULayer(choice_in_embed, self.n_hidden, mask_input=choice_mask_reshape,
                                                 grad_clipping=self.grad_clip)
        choice_backward = lasagne.layers.GRULayer(choice_in_embed, self.n_hidden, mask_input=choice_mask_reshape,
                                                  grad_clipping=self.grad_clip, backwards=True)
        choice_forward_slice = lasagne.layers.SliceLayer(choice_forward, -1, 1)
        choice_backward_slice = lasagne.layers.SliceLayer(choice_backward, 0, 1)
        choice_concat = lasagne.layers.ConcatLayer([choice_forward_slice, choice_backward_slice], axis=-1)
        choice_concat_reshape = lasagne.layers.ReshapeLayer(choice_concat, shape=(self.batch_size, self.choice_num, -1))

        analysis_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.analysis_num, self.max_seq_len))
        analysis_in_reshape = lasagne.layers.ReshapeLayer(analysis_in,
                                                          shape=(self.batch_size * self.analysis_num, self.max_seq_len))
        analysis_in_embed = EmbeddingLayer(analysis_in_reshape, input_size=self.vocab_size,
                                           output_size=self.embedding_size,
                                           W=self.index2vec)
        # dropout layer
        analysis_in_embed = lasagne.layers.DropoutLayer(analysis_in_embed, p=self.dropout_rate)
        analysis_mask = lasagne.layers.InputLayer(shape=(self.batch_size, self.analysis_num, self.max_seq_len))
        analysis_mask_reshape = lasagne.layers.ReshapeLayer(analysis_mask, shape=(
        self.batch_size * self.analysis_num, self.max_seq_len))
        analysis_forward = lasagne.layers.GRULayer(analysis_in_embed, self.n_hidden, mask_input=analysis_mask_reshape,
                                                   grad_clipping=self.grad_clip)
        analysis_backward = lasagne.layers.GRULayer(analysis_in_embed, self.n_hidden, mask_input=analysis_mask_reshape,
                                                    grad_clipping=self.grad_clip, backwards=True)
        analysis_forward_slice = lasagne.layers.SliceLayer(analysis_forward, -1, 1)
        analysis_backward_slice = lasagne.layers.SliceLayer(analysis_backward, 0, 1)
        analysis_concat = lasagne.layers.ConcatLayer([analysis_forward_slice, analysis_backward_slice], axis=-1)
        analysis_concat_reshape = lasagne.layers.ReshapeLayer(analysis_concat,
                                                              shape=(self.batch_size, self.analysis_num, -1))

        # Two method to compute attention ana and ques
        AttentionTypeLayer = DotAttentionLayer
        if self.attention_type == 'bilinear_attention':
            AttentionTypeLayer = BilinearAttentionLayer
        ques_ana_attention = AttentionTypeLayer([analysis_concat_reshape, ques_concat], self.n_hidden * 2)

        # Two method to merge ana and ques
        SumTypeLayer = SumLayer
        if self.sum_type == 'mlp_sum_layer':
            SumTypeLayer = MLPSumLayer
        sum_attention = SumTypeLayer([ques_ana_attention, ques_concat], self.n_hidden * 2)

        l_pred = BatchedDotLayer((choice_concat_reshape, sum_attention))
        self.network = l_pred

        if self.pre_trained:
            with np.load(self.model_file) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.network, param_values)
            logging.info("model loading success")

        # test
        pred_probs = lasagne.layers.get_output(self.network, {ques_in: Ques, ques_mask: Ques_mask, choice_in: Choice,
                                                              choice_mask: Choice_mask,
                                                              analysis_in: Analysis, analysis_mask: Analysis_mask},
                                               deterministic=True)
        pred_probs = T.nnet.softmax(pred_probs)
        pred = T.argmax(pred_probs, axis=1)

        self.predict = theano.function([Ques, Ques_mask, Choice, Choice_mask, Analysis, Analysis_mask],
                                       [pred, pred_probs])

        logging.info("building model done!")


    def build_model_use_analysis_a(self):
        '''
        build my model use two analysis encoder: one for attention, one for representation
        '''
        logging.info("building network ...")
        Ques = T.matrix('Ques', dtype='int64')
        Ques_mask = T.matrix('Ques_mask', dtype=theano.config.floatX)
        Choice = T.tensor3('Choice', dtype='int64')
        Choice_mask = T.tensor3('Choice_mask', dtype=theano.config.floatX)
        Analysis = T.tensor3('Analysis', dtype='int64')
        Analysis_mask = T.tensor3('Analysis_mask', dtype=theano.config.floatX)

        EmbeddingLayer = EmbeddingUnregularizeLayer
        if self.regularizable:
            EmbeddingLayer = lasagne.layers.EmbeddingLayer

        ques_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.max_seq_len))
        ques_in_embed = EmbeddingLayer(ques_in, input_size=self.vocab_size, output_size=self.embedding_size,
                                       W=self.index2vec)
        # dropout layer
        ques_in_embed = lasagne.layers.DropoutLayer(ques_in_embed, p=self.dropout_rate)

        ques_mask = lasagne.layers.InputLayer(shape=(self.batch_size, self.max_seq_len))
        ques_forward = lasagne.layers.GRULayer(ques_in_embed, self.n_hidden, mask_input=ques_mask,
                                               grad_clipping=self.grad_clip)
        ques_backward = lasagne.layers.GRULayer(ques_in_embed, self.n_hidden, mask_input=ques_mask,
                                                grad_clipping=self.grad_clip,
                                                backwards=True)
        ques_forward_slice = lasagne.layers.SliceLayer(ques_forward, -1, 1)
        ques_backward_slice = lasagne.layers.SliceLayer(ques_backward, 0, 1)
        ques_concat = lasagne.layers.ConcatLayer([ques_forward_slice, ques_backward_slice], axis=-1)

        choice_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.choice_num, self.max_seq_len))
        choice_in_reshape = lasagne.layers.ReshapeLayer(choice_in,
                                                        shape=(self.batch_size * self.choice_num, self.max_seq_len))
        choice_in_embed = EmbeddingLayer(choice_in_reshape, input_size=self.vocab_size, output_size=self.embedding_size,
                                         W=self.index2vec)
        # dropout layer
        choice_in_embed = lasagne.layers.DropoutLayer(choice_in_embed, p=self.dropout_rate)

        choice_mask = lasagne.layers.InputLayer(shape=(self.batch_size, self.choice_num, self.max_seq_len))
        choice_mask_reshape = lasagne.layers.ReshapeLayer(choice_mask,
                                                          shape=(self.batch_size * self.choice_num, self.max_seq_len))
        choice_forward = lasagne.layers.GRULayer(choice_in_embed, self.n_hidden, mask_input=choice_mask_reshape,
                                                 grad_clipping=self.grad_clip)
        choice_backward = lasagne.layers.GRULayer(choice_in_embed, self.n_hidden, mask_input=choice_mask_reshape,
                                                  grad_clipping=self.grad_clip, backwards=True)
        choice_forward_slice = lasagne.layers.SliceLayer(choice_forward, -1, 1)
        choice_backward_slice = lasagne.layers.SliceLayer(choice_backward, 0, 1)
        choice_concat = lasagne.layers.ConcatLayer([choice_forward_slice, choice_backward_slice], axis=-1)
        choice_concat_reshape = lasagne.layers.ReshapeLayer(choice_concat, shape=(self.batch_size, self.choice_num, -1))

        # for attention
        analysis_in_a = lasagne.layers.InputLayer(shape=(self.batch_size, self.analysis_num, self.max_seq_len))
        analysis_in_reshape_a = lasagne.layers.ReshapeLayer(analysis_in_a, shape=(
        self.batch_size * self.analysis_num, self.max_seq_len))
        analysis_in_embed_a = EmbeddingLayer(analysis_in_reshape_a, input_size=self.vocab_size,
                                             output_size=self.embedding_size,
                                             W=self.index2vec)
        # dropout layer
        analysis_in_embed_a = lasagne.layers.DropoutLayer(analysis_in_embed_a, p=self.dropout_rate)
        analysis_mask_a = lasagne.layers.InputLayer(shape=(self.batch_size, self.analysis_num, self.max_seq_len))
        analysis_mask_reshape_a = lasagne.layers.ReshapeLayer(analysis_mask_a, shape=(
        self.batch_size * self.analysis_num, self.max_seq_len))
        analysis_forward_a = lasagne.layers.GRULayer(analysis_in_embed_a, self.n_hidden,
                                                     mask_input=analysis_mask_reshape_a,
                                                     grad_clipping=self.grad_clip)
        analysis_backward_a = lasagne.layers.GRULayer(analysis_in_embed_a, self.n_hidden,
                                                      mask_input=analysis_mask_reshape_a,
                                                      grad_clipping=self.grad_clip, backwards=True)
        analysis_forward_slice_a = lasagne.layers.SliceLayer(analysis_forward_a, -1, 1)
        analysis_backward_slice_a = lasagne.layers.SliceLayer(analysis_backward_a, 0, 1)
        analysis_concat_a = lasagne.layers.ConcatLayer([analysis_forward_slice_a, analysis_backward_slice_a], axis=-1)
        analysis_concat_reshape_a = lasagne.layers.ReshapeLayer(analysis_concat_a,
                                                                shape=(self.batch_size, self.analysis_num, -1))

        # for representation
        analysis_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.analysis_num, self.max_seq_len))
        analysis_in_reshape = lasagne.layers.ReshapeLayer(analysis_in,
                                                          shape=(self.batch_size * self.analysis_num, self.max_seq_len))
        analysis_in_embed = EmbeddingLayer(analysis_in_reshape, input_size=self.vocab_size,
                                           output_size=self.embedding_size,
                                           W=self.index2vec)
        # dropout layer
        analysis_in_embed = lasagne.layers.DropoutLayer(analysis_in_embed, p=self.dropout_rate)
        analysis_mask = lasagne.layers.InputLayer(shape=(self.batch_size, self.analysis_num, self.max_seq_len))
        analysis_mask_reshape = lasagne.layers.ReshapeLayer(analysis_mask, shape=(
        self.batch_size * self.analysis_num, self.max_seq_len))
        analysis_forward = lasagne.layers.GRULayer(analysis_in_embed, self.n_hidden, mask_input=analysis_mask_reshape,
                                                   grad_clipping=self.grad_clip)
        analysis_backward = lasagne.layers.GRULayer(analysis_in_embed, self.n_hidden, mask_input=analysis_mask_reshape,
                                                    grad_clipping=self.grad_clip, backwards=True)
        analysis_forward_slice = lasagne.layers.SliceLayer(analysis_forward, -1, 1)
        analysis_backward_slice = lasagne.layers.SliceLayer(analysis_backward, 0, 1)
        analysis_concat = lasagne.layers.ConcatLayer([analysis_forward_slice, analysis_backward_slice], axis=-1)
        analysis_concat_reshape = lasagne.layers.ReshapeLayer(analysis_concat,
                                                              shape=(self.batch_size, self.analysis_num, -1))

        # Two method to compute attention ana and ques
        if self.attention_type == "dot_attention":
            ques_ana_attention = DotAttentionLayer_a([analysis_concat_reshape, ques_concat, analysis_concat_reshape_a])
        elif self.attention_type == 'bilinear_attention':
            ques_ana_attention = BilinearAttentionLayer_a(
                [analysis_concat_reshape, ques_concat, analysis_concat_reshape_a], self.n_hidden * 2)

        # Two method to merge ana and ques
        if self.sum_type == 'sum_layer':
            sum_attention = SumLayer([ques_ana_attention, ques_concat], self.n_hidden * 2)
        elif self.sum_type == 'mlp_sum_layer':
            sum_attention = MLPSumLayer([ques_ana_attention, ques_concat], self.n_hidden * 2)

        l_pred = BatchedDotLayer((choice_concat_reshape, sum_attention))
        self.network = l_pred

        if self.pre_trained:
            with np.load(self.model_file) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.network, param_values)
            logging.info("model loading success")

        # test
        pred_probs = lasagne.layers.get_output(self.network, {ques_in: Ques, ques_mask: Ques_mask, choice_in: Choice,
                                                              choice_mask: Choice_mask,
                                                              analysis_in: Analysis, analysis_mask: Analysis_mask,
                                                              analysis_in_a: Analysis, analysis_mask_a: Analysis_mask},
                                               deterministic=True)
        pred_probs = T.nnet.softmax(pred_probs)
        pred = T.argmax(pred_probs, axis=1)

        self.predict = theano.function([Ques, Ques_mask, Choice, Choice_mask, Analysis, Analysis_mask],
                                       [pred, pred_probs])

        logging.info("building model done!")


    def build_model_use_km(self):
        '''
        build model: knowledge matrix
        '''
        logging.info("building network ...")
        Ques = T.matrix('Ques', dtype='int64')
        Ques_mask = T.matrix('Ques_mask', dtype=theano.config.floatX)
        Choice = T.tensor3('Choice', dtype='int64')
        Choice_mask = T.tensor3('Choice_mask', dtype=theano.config.floatX)
        Analysis = T.matrix('Analysis', dtype=theano.config.floatX)
        # Analysis_mask = T.tensor2('Analysis_mask', dtype=theano.config.floatX)

        EmbeddingLayer = EmbeddingUnregularizeLayer
        if self.regularizable:
            EmbeddingLayer = lasagne.layers.EmbeddingLayer

        ques_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.max_seq_len))
        ques_in_embed = EmbeddingLayer(ques_in, input_size=self.vocab_size, output_size=self.embedding_size,
                                       W=self.index2vec)
        # dropout layer
        ques_in_embed = lasagne.layers.DropoutLayer(ques_in_embed, p=self.dropout_rate)

        ques_mask = lasagne.layers.InputLayer(shape=(self.batch_size, self.max_seq_len))
        ques_forward = lasagne.layers.GRULayer(ques_in_embed, self.n_hidden, mask_input=ques_mask,
                                               grad_clipping=self.grad_clip)
        ques_backward = lasagne.layers.GRULayer(ques_in_embed, self.n_hidden, mask_input=ques_mask,
                                                grad_clipping=self.grad_clip,
                                                backwards=True)
        ques_forward_slice = lasagne.layers.SliceLayer(ques_forward, -1, 1)
        ques_backward_slice = lasagne.layers.SliceLayer(ques_backward, 0, 1)
        ques_concat = lasagne.layers.ConcatLayer([ques_forward_slice, ques_backward_slice], axis=-1)

        choice_in = lasagne.layers.InputLayer(shape=(self.batch_size, self.choice_num, self.max_seq_len))
        choice_in_reshape = lasagne.layers.ReshapeLayer(choice_in,
                                                        shape=(self.batch_size * self.choice_num, self.max_seq_len))
        choice_in_embed = EmbeddingLayer(choice_in_reshape, input_size=self.vocab_size, output_size=self.embedding_size,
                                         W=self.index2vec)
        # dropout layer
        choice_in_embed = lasagne.layers.DropoutLayer(choice_in_embed, p=self.dropout_rate)

        choice_mask = lasagne.layers.InputLayer(shape=(self.batch_size, self.choice_num, self.max_seq_len))
        choice_mask_reshape = lasagne.layers.ReshapeLayer(choice_mask,
                                                          shape=(self.batch_size * self.choice_num, self.max_seq_len))
        choice_forward = lasagne.layers.GRULayer(choice_in_embed, self.n_hidden, mask_input=choice_mask_reshape,
                                                 grad_clipping=self.grad_clip)
        choice_backward = lasagne.layers.GRULayer(choice_in_embed, self.n_hidden, mask_input=choice_mask_reshape,
                                                  grad_clipping=self.grad_clip, backwards=True)
        choice_forward_slice = lasagne.layers.SliceLayer(choice_forward, -1, 1)
        choice_backward_slice = lasagne.layers.SliceLayer(choice_backward, 0, 1)
        choice_concat = lasagne.layers.ConcatLayer([choice_forward_slice, choice_backward_slice], axis=-1)
        choice_concat_reshape = lasagne.layers.ReshapeLayer(choice_concat, shape=(self.batch_size, self.choice_num, -1))

        analysis_in = lasagne.layers.InputLayer((self.cluster_size, self.embedding_size))
        analysis_embed = KMEmbeddingLayer(analysis_in, input_size=self.cluster_size, output_size=self.embedding_size, W=self.km)
        analysis_concat_reshape = lasagne.layers.ReshapeLayer(analysis_embed, shape=(1, self.cluster_size, self.embedding_size))

        # Two method to compute attention ana and ques
        AttentionTypeLayer = DotAttentionLayer
        if self.attention_type == 'bilinear_attention':
            AttentionTypeLayer = BilinearAttentionLayer
        ques_ana_attention = AttentionTypeLayer([analysis_concat_reshape, ques_concat], self.n_hidden * 2)

        # Two method to merge ana and ques
        SumTypeLayer = SumLayer
        if self.sum_type == 'mlp_sum_layer':
            SumTypeLayer = MLPSumLayer
        sum_attention = SumTypeLayer([ques_ana_attention, ques_concat], self.n_hidden * 2)

        l_pred = BatchedDotLayer((choice_concat_reshape, sum_attention))
        self.network = l_pred

        if self.pre_trained:
            with np.load(self.model_file) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.network, param_values)
            logging.info("model loading success")

        # test
        pred_probs = lasagne.layers.get_output(self.network, {ques_in: Ques, ques_mask: Ques_mask, choice_in: Choice,
                                                              choice_mask: Choice_mask,
                                                              analysis_in: Analysis},
                                               deterministic=True)
        pred_probs = T.nnet.softmax(pred_probs)
        pred = T.argmax(pred_probs, axis=1)

        self.predict = theano.function([Ques, Ques_mask, Choice, Choice_mask, Analysis],
                                       [pred, pred_probs])

        logging.info("building model done!")


    def predict_value_use_analysis(self, data, analysis):
        logging.info("predicting values...")
        ques, ques_mask, choice, choice_mask = data[0], data[1], data[2], data[3]
        ana, ana_mask = analysis[0], analysis[1]
        n_batch = len(ques) // self.batch_size
        count = 0
        result = []
        for batch in range(n_batch):
            temp = self.predict(ques[batch * self.batch_size: (batch + 1) * self.batch_size],
                                ques_mask[batch * self.batch_size: (batch + 1) * self.batch_size],
                                choice[batch * self.batch_size: (batch + 1) * self.batch_size],
                                choice_mask[batch * self.batch_size: (batch + 1) * self.batch_size],
                                ana[batch * self.batch_size: (batch + 1) * self.batch_size],
                                ana_mask[batch * self.batch_size: (batch + 1) * self.batch_size])

            logging.info(temp[1])
            result.append(temp[1][0])
            for i in range(len(temp[0])):
                if temp[0][i] == 0:
                    count += 1

        logging.info("predicting Done!")

        return result


    def make_vec_dict_file(self, vector_file=VECTOR_FILE, word2index_pkl_file=WORD2INDEX_PKL_FILE,
                           index2vec_npy_file=INDEX2VEC_NPY_FILE):
        '''
        convert word2vec into word2index and index2vec
        :param vector_file:
        :param word2index_pkl_file:
        :param index2vec_npy_file:
        '''

        logging.info('prepare vec2index and index2vec file...')

        reader = codecs.open(vector_file, 'r', 'utf-8')
        reader.readline()

        word2index_map = {}
        index2vec_array = np.zeros((self.vocab_size, self.embedding_size), dtype=theano.config.floatX)

        # vector_array[self.vocab_size] is for unk, set as a random vector.
        word2index_map[UNK] = 0
        index2vec_array[0] = np.random.normal(size=self.embedding_size).astype(theano.config.floatX)

        index = 1
        for line in reader:
            first_space_index = line.index(' ')

            vector = np.array([float(item) for item in line[first_space_index + 1:-1].split()]).astype(
                dtype=theano.config.floatX)
            index2vec_array[index] = vector

            word = line[:first_space_index]
            if word not in word2index_map:
                word2index_map[word] = index
            else:
                word2index_map[u'<' + word + '>'] = index
            index += 1
            if index % 100000 == 0:
                logging.info(str(index) + ' vector has finished.')

        if index != len(index2vec_array):
            logging.info("index != len(index2vec_array)")

        # save
        cPickle.dump(word2index_map, open(word2index_pkl_file, 'wb'))
        np.save(index2vec_npy_file, index2vec_array)
        logging.info('vec2index and index2vec file done!')


    def adapt_seq(self, seq, word2index_map):
        '''
		adapte seqlen to MAX_SEQ_LEN and generate seq_mask
		:param seq:
		:return:
		'''
        ada_seq = []
        mask = np.ones(self.max_seq_len)
        seq_len = len(seq)
        if (seq_len) > self.max_seq_len:
            for word in seq[-self.max_seq_len:]:
                index = 0  # index 0 is UNK
                if word in word2index_map:
                    index = word2index_map[word]
                ada_seq.append(index)
        else:
            for word in seq:
                index = 0
                if word in word2index_map:
                    index = word2index_map[word]
                ada_seq.append(index)
            for _ in range(seq_len, self.max_seq_len):
                ada_seq.append(0)
            mask[seq_len:] = 0.
        return np.array(ada_seq).astype('int64'), mask.astype(theano.config.floatX)


    def make_data2index_file(self, data_file, data2index_pkl_file):
        '''
		load the data, seperate Question and Choice, generate a dictionary, finally return word2index
		:param data_file: DATA_FILE
		:return: word2index
		'''
        logging.info('prepare data2index file...')
        file = codecs.open(data_file, 'r', encoding='utf-8')

        sentence = [line.strip().split() for line in file]
        problem_number = len(sentence) // 6

        ques = np.zeros((problem_number, self.max_seq_len)).astype('int64')
        ques_mask = np.zeros((problem_number, self.max_seq_len)).astype(theano.config.floatX)
        choice = np.zeros((problem_number, self.choice_num, self.max_seq_len)).astype('int64')
        choice_mask = np.zeros((problem_number, self.choice_num, self.max_seq_len)).astype(theano.config.floatX)
        for i in range(problem_number):
            ques_temp, ques_mask_temp = self.adapt_seq(np.concatenate(sentence[i * 6:i * 6 + 2], axis=0),
                                                       self.word2index_map)

            ques[i:] = ques_temp
            ques_mask[i:] = ques_mask_temp
            for j in range(self.choice_num):
                choice[i, j, :], choice_mask[i, j, :] = self.adapt_seq(sentence[i * 6 + j + 2], self.word2index_map)
            if (i + 1) % 2000 == 0:
                logging.info("{} data preparation done!".format((i + 1)))
        data = {'Q': ques, 'Q_mask': ques_mask, 'C': choice, 'C_mask': choice_mask}
        cPickle.dump(data, open(data2index_pkl_file, 'w'))

        logging.info('data2index file done!')


    def load_data(self, data2index_pkl_file):
        '''
		:return: train_data, valid_data, test_data
		'''
        logging.info("loading data ...")
        data = cPickle.load(open(data2index_pkl_file, 'r'))

        self.indices = np.arange(len(data['Q']))
        if self.shuffle:
            np.random.shuffle(self.indices)
            logging.info("data shuffle done!")

        train_data_size = int(self.train_data_rate * len(data['Q']))
        valid_data_size = int(self.valid_data_rate * len(data['Q']))

        ques_train = data['Q'][self.indices[:train_data_size]]
        ques_mask_train = data['Q_mask'][self.indices[:train_data_size]]
        choice_train = data['C'][self.indices[:train_data_size]]
        choice_mask_train = data['C_mask'][self.indices[:train_data_size]]

        ques_valid = data['Q'][self.indices[train_data_size:train_data_size + valid_data_size]]
        ques_mask_valid = data['Q_mask'][self.indices[train_data_size:train_data_size + valid_data_size]]
        choice_valid = data['C'][self.indices[train_data_size:train_data_size + valid_data_size]]
        choice_mask_valid = data['C_mask'][self.indices[train_data_size:train_data_size + valid_data_size]]

        ques_test = data['Q'][self.indices[train_data_size + valid_data_size:]]
        ques_mask_test = data['Q_mask'][self.indices[train_data_size + valid_data_size:]]
        choice_test = data['C'][self.indices[train_data_size + valid_data_size:]]
        choice_mask_test = data['C_mask'][self.indices[train_data_size + valid_data_size:]]

        self.train_data = (ques_train, ques_mask_train, choice_train, choice_mask_train)
        self.valid_data = (ques_valid, ques_mask_valid, choice_valid, choice_mask_valid)
        self.test_data = (ques_test, ques_mask_test, choice_test, choice_mask_test)

        logging.info("data loading done!")


    def make_analysis2index_file(self, ana_file, ana2index_pkl_file):
        '''
        convert analysis raw text into model input format
        '''
        logging.info('prepare ana2index file...')
        file = codecs.open(ana_file, 'r', encoding='utf-8')
        sentence = [line.strip().split() for line in file]

        problem_number = len(sentence) // self.analysis_num
        ana = np.zeros((problem_number, self.analysis_num, self.max_seq_len)).astype('int64')
        ana_mask = np.zeros((problem_number, self.analysis_num, self.max_seq_len)).astype(theano.config.floatX)
        for i in range(problem_number):
            for j in range(self.analysis_num):
                ana[i, j, :], ana_mask[i, j, :] = self.adapt_seq(sentence[i * self.analysis_num + j],
                                                                 self.word2index_map)
            if (i + 1) % 2000 == 0:
                logging.info("{} analysis preparation done!".format((i + 1)))
        analysis = {'A': ana, 'A_mask': ana_mask}
        cPickle.dump(analysis, open(ana2index_pkl_file, 'w'))

        logging.info('ana2index file done!')


    def load_analysis(self, ana2index_pkl_file):
        '''
        load analysis input format for model
        '''
        logging.info('loading analysis ...')
        analysis = cPickle.load(open(ana2index_pkl_file, 'r'))
        train_data_size = int(self.train_data_rate * len(analysis['A']))
        valid_data_size = int(self.valid_data_rate * len(analysis['A']))

        ana_train = analysis['A'][self.indices[:train_data_size]]
        ana_mask_train = analysis['A_mask'][self.indices[:train_data_size]]

        ana_valid = analysis['A'][self.indices[train_data_size: train_data_size + valid_data_size]]
        ana_mask_valid = analysis['A_mask'][self.indices[train_data_size: train_data_size + valid_data_size]]

        ana_test = analysis['A'][self.indices[train_data_size + valid_data_size:]]
        ana_mask_test = analysis['A_mask'][self.indices[train_data_size + valid_data_size:]]

        self.train_analysis = (ana_train, ana_mask_train)
        self.valid_analysis = (ana_valid, ana_mask_valid)
        self.test_analysis = (ana_test, ana_mask_test)

        logging.info('analysis loading done!')

    def load_km(self, km2vec_npy_file="data/gold/km.npy"):
        self.km = np.load(km2vec_npy_file)
        assert self.km.shape == (self.cluster_size, self.embedding_size)

    def load_word2index(self, word2index_pkl_file=WORD2INDEX_PKL_FILE):
        '''
        load word2index map into model
        '''
        logging.info("loading word2index ...")
        self.word2index_map = cPickle.load(open(word2index_pkl_file, 'r'))
        assert len(self.word2index_map) == self.vocab_size
        logging.info("word2index file loading done!")


    def load_index2vec(self, index2vec_npy_file=INDEX2VEC_NPY_FILE):
        '''
        load index2vec array into model
        '''
        logging.info("loading index2vec ...")
        self.index2vec = np.load(index2vec_npy_file)
        assert self.index2vec.shape == (self.vocab_size, self.embedding_size)
        logging.info("index2vec file loading done!")


def nn_pred(args):
    '''
    :param args: model parameter
    :return: all problem predict value
    '''
    np.random.seed(41)
    lasagne.random.set_rng(np.random.RandomState(1013))

    logging.info('*' * 50)

    model = Model(**args.__dict__)

    logging.info("use analysis to improve performance")
    if model.build_ac:
        logging.info("build_model_use_analysis_a")
        model.build_model_use_analysis_a()
    else:
        logging.info("build_model_use_analysis")
        model.build_model_use_analysis()

    logging.info("Predictiong gaokao data with analysis...")

    model.make_data2index_file(data_file=ROOT_DIR + "test_prob_seg.txt",
                               data2index_pkl_file=ROOT_DIR + 'test_prob_seg.data2index.pkl')
    gaokao_test_data = cPickle.load(open(ROOT_DIR + 'test_prob_seg.data2index.pkl', 'r'))
    gaokao_test_data = (
    gaokao_test_data['Q'], gaokao_test_data['Q_mask'], gaokao_test_data['C'], gaokao_test_data['C_mask'])

    model.make_analysis2index_file(ana_file=ROOT_DIR + "test_ana_seg.txt",
                                   ana2index_pkl_file=ROOT_DIR + 'test_ana_seg.ana2index.pkl')
    analysis_test_data = cPickle.load(open(ROOT_DIR + 'test_ana_seg.ana2index.pkl'))
    analysis_test_data = (analysis_test_data['A'], analysis_test_data['A_mask'])

    result = model.predict_value_use_analysis(gaokao_test_data, analysis_test_data)
    logging.info("-" * 50)
    return result

