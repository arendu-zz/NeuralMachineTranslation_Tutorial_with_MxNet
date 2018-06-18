#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
from src.utils import Corpus
from src.models import RNNLM
import mxnet as mx
from mxnet import nd, autograd, gluon
import random

if __name__ == '__main__':
    random.seed(1234)
    mx.random.seed(1234)
    opt= argparse.ArgumentParser(description="write program description here")
    #insert options here
    opt.add_argument('-t', action='store' , dest='train_file', default='./data/translation/train.en')
    opt.add_argument('--embedding_size', action='store', dest='embedding_size', default=100, type=int)
    opt.add_argument('--hidden_size', action='store', dest='hidden_size', default=100, type=int)
    opt.add_argument('--num_layers', action='store', dest='num_layers', default=1, type=int)
    options = opt.parse_args()
    ctx = mx.cpu(0)

    corpus = Corpus(options.train_file)
    model = RNNLM(vocab_size=len(corpus.dictionary),
                  embedding_size=options.embedding_size,
                  hidden_size=options.hidden_size,
                  num_layers=options.num_layers)
    print(model)
    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    softmax_ce_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True, #expects int/categorical labels
                                              from_logits=False, #expects unnormalized numbers
                                              batch_axis=0)
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 0.1})
    batch_size = 1
    log_interval = 20
    clip=1.0
    epochs = 2
    for e in range(epochs):
        total_L = 0
        interval_L = 0
        random.shuffle(corpus.numberized_train)
        for idx, numberized_sent in enumerate(corpus.numberized_train):
            x = numberized_sent[:-1]
            x = x.reshape(x.shape[0], batch_size)
            y = numberized_sent[1:]
            x = x.as_in_context(ctx)
            y = y.as_in_context(ctx)
            with autograd.record():
                output, hidden_states, final_states = model(x)
                loss = softmax_ce_loss(output, y)
            loss.backward()
            grads = [i.grad(ctx) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, clip)
            trainer.step(batch_size)
            interval_L += nd.mean(loss).asscalar()
            total_L += nd.mean(loss).asscalar()
            if idx % log_interval == 0 and idx > 0:
                print("%s (%s / %s), loss: %s" % (e, idx, len(corpus.numberized_train), interval_L/ log_interval))
                interval_L = 0
        print("Train Epoch %s, loss: %s" % (e, total_L/ len(corpus.numberized_train)))
