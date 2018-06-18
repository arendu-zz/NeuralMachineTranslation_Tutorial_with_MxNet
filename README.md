# jsalt2018-nmt-lab

Install mxnet and all the required dependencies with the following commands:
```
$>git clone http://github.com/szha/gluon-crash-course -b jsalt
$>cd gluon-crash-course
$>conda env create -f env/environment.yml 
```

Once you complete filling in all the `TODOs` in `src.models.RNNLM` run the command below and check if you match this output log:
```
python lm.py -t ./data/words/train.es
RNNLM(
  (dropout): Dropout(p = 0.0, axes=())
  (embedding): Embedding(40 -> 100, float32)
  (rnn): RNN(100 -> 100, TNC)
  (output): Dense(100 -> 40, linear)
)
0 (20 / 84316), loss: 3.726979446411133
0 (40 / 84316), loss: 3.1106373190879824
0 (60 / 84316), loss: 3.1474567532539366
0 (80 / 84316), loss: 3.034141170978546
0 (100 / 84316), loss: 2.933252203464508
0 (120 / 84316), loss: 2.949366736412048
0 (140 / 84316), loss: 2.9076883435249328
0 (160 / 84316), loss: 2.8319573283195494
0 (180 / 84316), loss: 2.8311381936073303
0 (200 / 84316), loss: 2.818974471092224
0 (220 / 84316), loss: 2.8247193574905394
0 (240 / 84316), loss: 2.8505154728889464
0 (260 / 84316), loss: 2.810903549194336
0 (280 / 84316), loss: 2.80368115901947
0 (300 / 84316), loss: 2.7881394267082213
```

Now try increasing the number of layers in your RNNLM:
```
$>python lm.py -t ./data/words/train.es --num_layers 3
RNNLM(
  (dropout): Dropout(p = 0.3, axes=())
  (embedding): Embedding(40 -> 100, float32)
  (rnn): RNN(100 -> 100, TNC, num_layers=3, dropout=0.3)
  (output): Dense(100 -> 40, linear)
)
0 (20 / 84316), loss: 3.783879339694977
0 (40 / 84316), loss: 3.26540789604187
0 (60 / 84316), loss: 3.222878885269165
0 (80 / 84316), loss: 3.2052329659461973
0 (100 / 84316), loss: 3.106749212741852
0 (120 / 84316), loss: 3.0908856272697447
0 (140 / 84316), loss: 3.034270131587982
0 (160 / 84316), loss: 2.9749829173088074
0 (180 / 84316), loss: 2.9491508483886717
0 (200 / 84316), loss: 2.9033187508583067
0 (220 / 84316), loss: 2.896671462059021
0 (240 / 84316), loss: 2.9300551533699037
0 (260 / 84316), loss: 2.918925333023071
0 (280 / 84316), loss: 2.8628843784332276
0 (300 / 84316), loss: 2.909125304222107
```

Next you will implement a simple encoder-decoder without attention.
Complete the scaffolding code in `src.models.EncoderDecoder` and run the command below:
```
python encoder_decoder.py --src_train ./data/words/train.es --tgt_train ./data/words/train.pt
EncoderDecoder(
  (encoder): RNNLM(
    (dropout): Dropout(p = 0.0, axes=())
    (embedding): Embedding(40 -> 100, float32)
    (rnn): RNN(100 -> 100, TNC)
    (output): Dense(100 -> 40, linear)
  )
  (decoder): RNNLM(
    (dropout): Dropout(p = 0.0, axes=())
    (embedding): Embedding(40 -> 100, float32)
    (rnn): RNN(100 -> 100, TNC)
    (output): Dense(100 -> 40, linear)
  )
)
0 (20 / 84316), loss: 3.5951297521591186
0 (40 / 84316), loss: 2.9892606258392336
0 (60 / 84316), loss: 2.9440046548843384
0 (80 / 84316), loss: 2.8132421135902406
0 (100 / 84316), loss: 2.8034116506576536
0 (120 / 84316), loss: 2.752419650554657
0 (140 / 84316), loss: 2.664055275917053
0 (160 / 84316), loss: 2.5977513015270235
0 (180 / 84316), loss: 2.651363104581833
0 (200 / 84316), loss: 2.5822153210639955
0 (220 / 84316), loss: 2.6784459233283995
0 (240 / 84316), loss: 2.529151886701584
0 (260 / 84316), loss: 2.6145973801612854
0 (280 / 84316), loss: 2.6833040833473207
0 (300 / 84316), loss: 2.4839312553405763
```

Now increase the number of layers:
```
python encoder_decoder.py --src_train ./data/words/train.es --tgt_train ./data/words/train.pt --num_layers 3
EncoderDecoder(
  (encoder): RNNLM(
    (dropout): Dropout(p = 0.0, axes=())
    (embedding): Embedding(40 -> 100, float32)
    (rnn): RNN(100 -> 100, TNC, num_layers=3)
    (output): Dense(100 -> 40, linear)
  )
  (decoder): RNNLM(
    (dropout): Dropout(p = 0.0, axes=())
    (embedding): Embedding(40 -> 100, float32)
    (rnn): RNN(100 -> 100, TNC, num_layers=3)
    (output): Dense(100 -> 40, linear)
  )
)
0 (20 / 84316), loss: 4.082468640804291
0 (40 / 84316), loss: 3.1678009986877442
0 (60 / 84316), loss: 3.1164113759994505
0 (80 / 84316), loss: 3.1208514094352724
0 (100 / 84316), loss: 3.0065815687179565
0 (120 / 84316), loss: 3.0014429450035096
0 (140 / 84316), loss: 2.9589979529380797
0 (160 / 84316), loss: 2.8412908554077148
0 (180 / 84316), loss: 2.830174469947815
0 (200 / 84316), loss: 2.8452255010604857
0 (220 / 84316), loss: 2.898094516992569
0 (240 / 84316), loss: 2.787045431137085
0 (260 / 84316), loss: 2.772556495666504
0 (280 / 84316), loss: 3.1168592810630797
0 (300 / 84316), loss: 2.8500282645225523
```

Finally, you are ready to implement an encoder-decoder with attention.
Complete the scaffolding code in `src.models.EncoderDecoderAttention` and run the command below:
```
python encoder_decoder.py --src_train ./data/words/train.es --tgt_train ./data/words/train.pt --attn
EncoderDecoderAttention(
  (dropout): Dropout(p = 0.0, axes=())
  (encoder): RNN(100 -> 100, TNC, bidirectional)
  (src_embedding): Embedding(40 -> 100, float32)
  (attention): Dense(300 -> 1, linear)
  (decoder_cell): SequentialRNNCell(
  (0): RNNCell(300 -> 100, tanh)
  )
  (output): Dense(100 -> 40, linear)
  (tgt_embedding): Embedding(40 -> 100, float32)
)
0 (20 / 84316), loss: 3.831316685676575
0 (40 / 84316), loss: 3.3857186913490294
0 (60 / 84316), loss: 3.2950724005699157
0 (80 / 84316), loss: 3.1895662546157837
0 (100 / 84316), loss: 3.1972587585449217
0 (120 / 84316), loss: 3.194217872619629
0 (140 / 84316), loss: 3.0651586174964907
0 (160 / 84316), loss: 3.0520243644714355
0 (180 / 84316), loss: 3.0168715000152586
0 (200 / 84316), loss: 3.0374971508979796
0 (220 / 84316), loss: 2.984034526348114
0 (240 / 84316), loss: 2.9802499413490295
0 (260 / 84316), loss: 2.8145950198173524
0 (280 / 84316), loss: 3.0609739780426026
0 (300 / 84316), loss: 2.9290491223335264
```

Again, your implementation should support increasing the number of layers, run the command below:
```
python encoder_decoder.py --src_train ./data/words/train.es --tgt_train ./data/words/train.pt --attn --num_layers 3
EncoderDecoderAttention(
  (dropout): Dropout(p = 0.0, axes=())
  (encoder): RNN(100 -> 100, TNC, num_layers=3, bidirectional)
  (src_embedding): Embedding(40 -> 100, float32)
  (attention): Dense(300 -> 1, linear)
  (decoder_cell): SequentialRNNCell(
  (0): RNNCell(300 -> 100, tanh)
  (1): RNNCell(None -> 100, tanh)
  (2): RNNCell(None -> 100, tanh)
  )
  (output): Dense(100 -> 40, linear)
  (tgt_embedding): Embedding(40 -> 100, float32)
)
0 (20 / 84316), loss: 4.183947658538818
0 (40 / 84316), loss: 3.7976392269134522
0 (60 / 84316), loss: 3.653985357284546
0 (80 / 84316), loss: 3.784492886066437
0 (100 / 84316), loss: 3.4958001017570495
0 (120 / 84316), loss: 3.5960230469703673
0 (140 / 84316), loss: 3.4644107460975646
0 (160 / 84316), loss: 3.34350346326828
0 (180 / 84316), loss: 3.3598593950271605
0 (200 / 84316), loss: 3.4301453948020937
0 (220 / 84316), loss: 3.3763737201690676
0 (240 / 84316), loss: 3.4125502824783327
0 (260 / 84316), loss: 3.345878207683563
0 (280 / 84316), loss: 3.414934813976288
0 (300 / 84316), loss: 3.272854042053223
```
