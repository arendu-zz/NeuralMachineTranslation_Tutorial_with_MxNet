#!/usr/bin/env python
__author__ = 'arenduchintala'
import mxnet as mx
from mxnet import nd

SPL_SYMS=['<BOS>', '<EOS>', '<UNK>']

class Dictionary(object):
    def __init__(self,):
        self.char2count = {}
        self.char2idx = {}
        self.idx2char = {}
        for spl in SPL_SYMS:
            self.add_char(spl)

    def count_char(self,c):
        self.char2count[c] = self.char2count.get(c, 0) + 1

    def add_char(self, c):
        self.char2idx[c] = self.char2idx.get(c, len(self.char2idx))
        self.idx2char[self.char2idx[c]] = c
        return self.char2idx[c]
    
    def get_charid(self, c):
        return self.char2idx.get(c, self.char2idx['<UNK>'])

    def __len__(self,):
        return len(self.char2idx)


class Corpus(object):
    def __init__(self, train_file, dev_file=None):
        self.dictionary = self.init_dictionary(train_file)
        self.numberized_train = self.numberize(train_file)
        if dev_file is not None:
            self.numberized_dev = self.numberize(dev_file)
        else:
            self.numberized_dev = []

    def init_dictionary(self, train_file, min_count=0):
        d = Dictionary()
        with open(train_file, 'r', encoding='utf-8') as corpus:
            for line in corpus:
                for char in line.strip().split():
                    d.count_char(char)
        for char,count in d.char2count.items():
            if count >= min_count:
                d.add_char(char)
            else:
                pass
        return d

    def numberize(self, corpus_file):
        numberized_corpus = []
        with open(corpus_file, 'r', encoding='utf-8') as corpus:
            for line in corpus:
                numberized_line = [self.dictionary.get_charid('<BOS>')]
                for char in line.strip().split():
                    numberized_line.append(self.dictionary.get_charid(char))
                numberized_line.append(self.dictionary.get_charid('<EOS>'))
                numberized_line = mx.nd.array(numberized_line, dtype='int32')
                numberized_corpus.append(numberized_line)
        return numberized_corpus
