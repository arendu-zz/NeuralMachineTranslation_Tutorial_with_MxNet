# jsalt2018-nmt-lab

Install mxnet and all the required dependencies with the following commands:
```
$>git clone http://github.com/szha/gluon-crash-course -b jsalt
$>cd gluon-crash-course
$>conda env create -f env/environment.yml 
```

```
$>python lm.py -t ./data/words/train.es
RNNLM(
  (dropout): Dropout(p = 0.3, axes=())
  (embedding): Embedding(40 -> 100, float32)
  (rnn): RNN(100 -> 100, TNC, dropout=0.3)
  (output): Dense(100 -> 40, linear)
)
0 (20 / 84316), loss: 3.7949697732925416
0 (40 / 84316), loss: 3.2603787064552305
0 (60 / 84316), loss: 3.208081007003784
0 (80 / 84316), loss: 3.0942046999931336
0 (100 / 84316), loss: 3.044014883041382
0 (120 / 84316), loss: 3.05588299036026
0 (140 / 84316), loss: 3.0130700945854185
0 (160 / 84316), loss: 2.885222148895264
0 (180 / 84316), loss: 2.920358407497406
0 (200 / 84316), loss: 2.9004335522651674
0 (220 / 84316), loss: 2.884094536304474
0 (240 / 84316), loss: 2.9174814581871034
0 (260 / 84316), loss: 2.8341264367103576
0 (280 / 84316), loss: 2.8890059113502504
0 (300 / 84316), loss: 2.8589306831359864
```
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
```
$>python encoder_decoder.py --src_train ./data/words/train.es --tgt_train ./data/words/train.pt
EncoderDecoder(
  (dropout): Dropout(p = 0.3, axes=())
  (encoder): RNNLM(
    (dropout): Dropout(p = 0.3, axes=())
    (embedding): Embedding(40 -> 100, float32)
    (rnn): RNN(100 -> 100, TNC, dropout=0.3)
    (output): Dense(100 -> 40, linear)
  )
  (decoder): RNNLM(
    (dropout): Dropout(p = 0.3, axes=())
    (embedding): Embedding(40 -> 100, float32)
    (rnn): RNN(100 -> 100, TNC, dropout=0.3)
    (output): Dense(100 -> 40, linear)
  )
)
0 (20 / 84316), loss: 3.8579011797904967
0 (40 / 84316), loss: 3.1551016569137573
0 (60 / 84316), loss: 3.0131725430488587
0 (80 / 84316), loss: 2.8448883295059204
0 (100 / 84316), loss: 2.800902581214905
0 (120 / 84316), loss: 2.876548981666565
0 (140 / 84316), loss: 2.741378128528595
0 (160 / 84316), loss: 2.68673797249794
0 (180 / 84316), loss: 2.6842251479625703
0 (200 / 84316), loss: 2.650237166881561
0 (220 / 84316), loss: 2.629344326257706
0 (240 / 84316), loss: 2.7226999700069427
0 (260 / 84316), loss: 2.7118025183677674
0 (280 / 84316), loss: 2.681566244363785
0 (300 / 84316), loss: 2.6075175046920775
```
```
$>python encoder_decoder.py --src_train ./data/words/train.es --tgt_train ./data/words/train.pt --num_layers 3
EncoderDecoder(
  (dropout): Dropout(p = 0.3, axes=())
  (encoder): RNNLM(
    (dropout): Dropout(p = 0.3, axes=())
    (embedding): Embedding(40 -> 100, float32)
    (rnn): RNN(100 -> 100, TNC, num_layers=3, dropout=0.3)
    (output): Dense(100 -> 40, linear)
  )
  (decoder): RNNLM(
    (dropout): Dropout(p = 0.3, axes=())
    (embedding): Embedding(40 -> 100, float32)
    (rnn): RNN(100 -> 100, TNC, num_layers=3, dropout=0.3)
    (output): Dense(100 -> 40, linear)
  )
)
0 (20 / 84316), loss: 3.711759054660797
0 (40 / 84316), loss: 3.2243748188018797
0 (60 / 84316), loss: 3.193679964542389
0 (80 / 84316), loss: 3.2084130167961122
0 (100 / 84316), loss: 3.077763032913208
0 (120 / 84316), loss: 3.107494020462036
0 (140 / 84316), loss: 3.0566675662994385
0 (160 / 84316), loss: 2.9836545467376707
0 (180 / 84316), loss: 2.989021599292755
0 (200 / 84316), loss: 2.9571225166320803
0 (220 / 84316), loss: 3.5385950446128844
0 (240 / 84316), loss: 2.939542090892792
0 (260 / 84316), loss: 3.017833244800568
0 (280 / 84316), loss: 3.033288025856018
0 (300 / 84316), loss: 2.926031804084778
```
```
$>python encoder_decoder.py --src_train ./data/words/train.es --tgt_train ./data/words/train.pt  --attn
EncoderDecoderAttention(
  (dropout): Dropout(p = 0.3, axes=())
  (encoder): RNN(100 -> 100, TNC, dropout=0.3, bidirectional)
  (src_embedding): Embedding(40 -> 100, float32)
  (attention): Dense(300 -> 1, linear)
  (decoder_cell): SequentialRNNCell(
  (0): RNNCell(300 -> 100, tanh)
  )
  (output): Dense(100 -> 40, linear)
  (tgt_embedding): Embedding(40 -> 100, float32)
)
0 (20 / 84316), loss: 3.8282788753509522
0 (40 / 84316), loss: 3.4499674081802367
0 (60 / 84316), loss: 3.339201867580414
0 (80 / 84316), loss: 3.287199306488037
0 (100 / 84316), loss: 3.1413058161735536
0 (120 / 84316), loss: 3.2493163704872132
0 (140 / 84316), loss: 3.0453745245933534
0 (160 / 84316), loss: 3.025236999988556
0 (180 / 84316), loss: 3.066582727432251
0 (200 / 84316), loss: 3.0194608092308046
0 (220 / 84316), loss: 2.988750624656677
0 (240 / 84316), loss: 3.002424085140228
0 (260 / 84316), loss: 2.8438608407974244
0 (280 / 84316), loss: 3.096175467967987
0 (300 / 84316), loss: 2.901057040691376
```
```
$>python encoder_decoder.py --src_train ./data/words/train.es --tgt_train ./data/words/train.pt --num_layers 3 --attn
EncoderDecoderAttention(
  (dropout): Dropout(p = 0.3, axes=())
  (encoder): RNN(100 -> 100, TNC, num_layers=3, dropout=0.3, bidirectional)
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
0 (20 / 84316), loss: 4.1549996614456175
0 (40 / 84316), loss: 3.637652885913849
0 (60 / 84316), loss: 3.720323085784912
0 (80 / 84316), loss: 3.7175458431243897
0 (100 / 84316), loss: 3.4328295469284056
0 (120 / 84316), loss: 3.535549581050873
0 (140 / 84316), loss: 3.4518826484680174
0 (160 / 84316), loss: 3.3856226563453675
0 (180 / 84316), loss: 3.275982987880707
0 (200 / 84316), loss: 3.345855104923248
0 (220 / 84316), loss: 3.331212842464447
0 (240 / 84316), loss: 3.3261744499206545
0 (260 / 84316), loss: 3.2420976638793944
0 (280 / 84316), loss: 3.337713199853897
0 (300 / 84316), loss: 3.2777873396873476
```
