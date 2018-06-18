#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
from src.utils import Corpus
from src.models import EncoderDecoder, EncoderDecoderAttention
import mxnet as mx
from mxnet import nd, autograd, gluon
import random

if __name__ == '__main__':
    random.seed(1234)
    mx.random.seed(1234)
    opt= argparse.ArgumentParser(description="write program description here")
    #insert options here
    opt.add_argument('--src_train', action='store' , dest='src_train', default='./data/translation/train.en')
    opt.add_argument('--tgt_train', action='store' , dest='tgt_train', default='./data/translation/train.fr')
    opt.add_argument('--embedding_size', action='store', dest='embedding_size', default=100, type=int)
    opt.add_argument('--hidden_size', action='store', dest='hidden_size', default=100, type=int)
    opt.add_argument('--num_layers', action='store', dest='num_layers', default=1, type=int)
    opt.add_argument('--attn', action='store_true' ,dest='use_attention', default=False)
    options = opt.parse_args()
    ctx = mx.cpu(0)
    src_corpus = Corpus(options.src_train)
    tgt_corpus = Corpus(options.tgt_train)

    if options.use_attention:
        model = EncoderDecoderAttention(src_vocab_size=len(src_corpus.dictionary),
                               tgt_vocab_size=len(tgt_corpus.dictionary),
                               embedding_size=options.embedding_size,
                               hidden_size=options.hidden_size,
                               num_layers=options.num_layers)
    else:
        model = EncoderDecoder(src_vocab_size=len(src_corpus.dictionary),
                               tgt_vocab_size=len(tgt_corpus.dictionary),
                               embedding_size=options.embedding_size,
                               hidden_size=options.hidden_size,
                               num_layers=options.num_layers)
    print(model)
    model.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    softmax_ce_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=True, #expects int/categorical labels
                                              from_logits=False, #expects unnormalized numbers
                                              batch_axis=0)
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': 1.0})
    batch_size = 1
    log_interval = 20
    clip=1.0
    epochs = 2
    instance_idxs = list(range(len(src_corpus.numberized_train)))
    for e in range(epochs):
        total_L = 0.
        interval_L = 0.
        random.shuffle(instance_idxs)
        for idx, i in enumerate(instance_idxs):
            src_input = src_corpus.numberized_train[i]
            src_labels = src_input[1:]
            tgt = tgt_corpus.numberized_train[i]
            tgt_input = tgt[:-1]
            tgt_labels = tgt[1:]
            src_input = src_input.reshape(src_input.shape[0], batch_size)
            tgt_input = tgt_input.reshape(tgt_input.shape[0], batch_size)
            src_input = src_input.as_in_context(ctx)
            tgt_input = tgt_input.as_in_context(ctx)
            with autograd.record():
                tgt_outputs = model(src_input, tgt_input)
                loss = softmax_ce_loss(tgt_outputs, tgt_labels)
            loss.backward()
            grads = [i.grad(ctx) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, clip)
            trainer.step(batch_size, ignore_stale_grad=True)
            interval_L += nd.mean(loss).asscalar()
            total_L += nd.mean(loss).asscalar()
            if idx % log_interval == 0 and idx > 0:
                print("%s (%s / %s), loss: %s" % (e, idx, len(src_corpus.numberized_train), interval_L/ log_interval))
                interval_L = 0
        print("Train Epoch %s, loss: %s" % (e, total_L/ len(src_corpus.numberized_train)))




