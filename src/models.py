#!/usr/bin/env python
__author__ = 'arenduchintala'
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, rnn

class RNNLM(gluon.Block):
    def __init__(self,
                 vocab_size,
                 embedding_size=100,
                 hidden_size=100,
                 num_layers=1,
                 dropout=0.3, **kwargs):
        super(RNNLM, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        with self.name_scope():
            #TODO: create an embedding layer here 
            #TODO: the embedding layer takes a sequence of ints and converts them into a sequence of real-valued vectors
            #TODO: refer to https://mxnet.incubator.apache.org/api/python/gluon/nn.html 

            #TODO: create a RNN layer here
            #TODO: refer to documentation here https://mxnet.incubator.apache.org/api/python/gluon/rnn.html#mxnet.gluon.rnn.RNN

            #TODO: create a DENSE layer (decoder)
            #TODO: https://mxnet.incubator.apache.org/api/python/gluon/nn.html#mxnet.gluon.nn.Dense
            pass

    def forward(self, x, init_hidden_state=None):
        """
        Forward computation for EncoderDecoderAttention

        Parameters:
        x (mx.ndarray): source sequence including the <EOS> symbol
        init_hidden_state (mx.ndarray or None): if not None this should be used as the initial hidden state of the rnn
        """
        #TODO: use the embedding layer to convert x into a sequence of vector representations
        #emb should be (seq_len, batch_size, embedding_size)
        #TODO: use rnn over the vector representations
        # you might want to separately handle when init_hidden_state is None and when its not None
        #TODO: shape the output of the rnn layer to "fit" with the Dense layer
        #TODO: https://mxnet.incubator.apache.org/api/python/gluon/nn.html#mxnet.gluon.nn.Dense
        return outputs, hidden_states, final_states


class EncoderDecoder(gluon.Block):
    def __init__(self,
                src_vocab_size,
                tgt_vocab_size,
                embedding_size=100,
                hidden_size=100,
                num_layers=1,
                dropout=0.3, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        with self.name_scope():
            #TODO: create an encoder RNNLM 
            #TODO: create a decoder RNNLM
            pass

    def forward(self, x, y):
        """
        Forward computation for EncoderDecoder

        Parameters:
        x (mx.ndarray): source sequence including the <EOS> symbol
        y (mx.ndarray): target sequence excluding the <EOS> symbol
        """
        #TODO: "encode" the source sequence by passing the input x through the encoder RNNLM.
        #TODO: decode the target sequence using the decoder AND the last hidden state from the encoder.
        return out_tgt

class EncoderDecoderAttention(gluon.Block):
    def __init__(self,
                src_vocab_size,
                tgt_vocab_size,
                embedding_size=100,
                hidden_size=100,
                num_layers=1,
                dropout=0.3, **kwargs):
        super(EncoderDecoderAttention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.context_size = 2 * hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        with self.name_scope():
            #TODO: create an encoder RNN
            #unlike the RNNLM and EncoderDecoder model this time the encoder should be bidirectional
            #TODO: create a source side embedding layer
            #TODO: create dense layer for computing a single attention weight

            #TODO: create a "decoder cell" for the target side decoder
            #TODO: we have to use a RNNCell for this implementation 
            #TODO: refer to https://mxnet.incubator.apache.org/api/python/gluon/rnn.html
            #TODO: if you want to support multiple layers in the decoder then you should refer https://mxnet.incubator.apache.org/api/python/gluon/rnn.html#mxnet.gluon.rnn.SequentialRNNCell

            #TODO: create a dense layer which produces an unnormalized distribution over target vocabulary

            #TODO: create a target side embedding layer
            pass

    def forward(self, x, y):
        """
        Forward computation for EncoderDecoderAttention

        Parameters:
        x (mx.ndarray): source sequence including the <EOS> symbol
        y (mx.ndarray): target sequence excluding the <EOS> symbol
        """
        #TODO: use the source embedding to convert the source sentence into a vector sequence

        #TODO: use the source encoder to "encode" the vector sequence and get a sequence of hidden states

        
        #TODO: initialize a zero ndarray to represent the initial state of the previous time-step target side hidden state
        #TODO: if you want to use more than one layer you may want to use "begin_state" functionality https://mxnet.incubator.apache.org/api/python/gluon/rnn.html#mxnet.gluon.rnn.RecurrentCell.begin_state

        #TODO: initialize a zero ndarray to represent the initial embedding of the previous time-step target output
        
        #a ouputs list which will store unnormalized outputs for each target side word
        outputs = []
        # loop over each word in the target sequence
        for j, curr_vocab in enumerate(y):
            #TODO: use the get_attention_weights function and get a set of attention weights for the j'th target word
            #TODO: the input to get_attention_weights is the sequence of source-side hidden states and the previous target-side hidden state

            #TODO: using the attention weights and the source-side hidden states get the context vector using the "get_context_vector" function.

            #TODO:Use the decoder cell to get the current target-side hidden state
            #TODO:The inputs for the decoder cell are (i) the context vector concated with  previous target word embedding and (ii) the previous target-side hidden state

            #TODO: get the unnormalized output using the current target-side hidden state and the output layer
            #TODO: append the unnormalized output into the outputs list
            
            #TODO: set the previous target-side hidden state as the current target-side hidden state (in preparation for the next iteration through this loop)
            #TODO: if your decoder was multi-layer you should use all the internal hidden_states

            #TODO: set the previous target embedding using the current target-side word (again in preparation for the next iteration through this loop)
            pass
        
        outputs = mx.nd.concat(*outputs, dim=0)
        return outputs

    def get_attention_weights(self, src_hidden_states, prev_tgt_hidden_state):
        """
        As the name suggests this function returns a sequence of weights which will be used to compute the context vector

        Parameters:
        src_hidden_states (mx.ndarray): a sequence of source-side hidden states
        prev_tgt_hidden_state (mx.ndarray): the vector representing the previous time-step target side hidden state
        """
        #a buffer to store the unnormalized attention weights
        weights = [] 
        for i, src_hidden_state in enumerate(src_hidden_states):
            #TODO: use the attention layer to obtain a single attention weight for each source-side hidden state

            #TODO: add the attention weight into the buffer
            pass

        #TODO: conver the weights list into a ndarray and normalize it using a softmax function.
        #TODO: refer to https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html?highlight=softmax#mxnet.ndarray.NDArray.softmax
        return normalized_weights

    def get_context_vector(self, weights, src_hidden_states):
        """
        Performs the weighted avg. over the sequence of hidden states

        Parameters:
        weights (mx.ndarray): a sequence of attention weights 
        src_hidden_state (mx.ndarray): a sequence of source-side hidden states
        """
        #TODO: take the weighted avg of the src_hidden_states
        #TODO: you may find this utilities helpful https://mxnet.incubator.apache.org/api/python/ndarray/ndarray.html#expanding-array-elements
        return context_vector
