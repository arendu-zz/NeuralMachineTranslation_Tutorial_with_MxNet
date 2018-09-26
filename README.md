# jsalt2019-nmt-lab

### Setup:
Install mxnet and all the required dependencies with the following commands:
```
$>git clone http://github.com/szha/gluon-crash-course -b jsalt
$>cd gluon-crash-course
$>conda env create -f env/environment.yml 
```
### RNN Language Model:
Once you complete filling in all the `TODOs` in `src.models.RNNLM` run the command below and check if you match this output log:
```
$>python lm.py
Train Epoch 0, loss: 3.2933573365211486
Train Epoch 1, loss: 2.947761654853821
Train Epoch 2, loss: 2.7929221391677856
Train Epoch 3, loss: 2.6799631357192992
Train Epoch 4, loss: 2.5977691531181337
Train Epoch 5, loss: 2.5197918355464934
Train Epoch 6, loss: 2.429540681838989
Train Epoch 7, loss: 2.34314603805542
Train Epoch 8, loss: 2.2721444010734557
Train Epoch 9, loss: 2.1963339030742643
Train Epoch 10, loss: 2.1091691732406614
```
Now try increasing the number of layers in your RNNLM:
```
$>python lm.py --num_layers 3
Train Epoch 0, loss: 3.151236379146576
Train Epoch 1, loss: 2.863504505157471
Train Epoch 2, loss: 2.782533419132233
Train Epoch 3, loss: 2.676173996925354
Train Epoch 4, loss: 2.5878071546554566
Train Epoch 5, loss: 2.4914966344833376
Train Epoch 6, loss: 2.390135943889618
Train Epoch 7, loss: 2.3071948528289794
Train Epoch 8, loss: 2.199465584754944
Train Epoch 9, loss: 2.0951857686042787
Train Epoch 10, loss: 2.011309826374054
```
### Encoder-Decoder Model for translation:
Next you will implement a simple encoder-decoder without attention.
Complete the scaffolding code in `src.models.EncoderDecoder` and run the command below:
```
$>python encoder_decoder.py
0 (5 / 20), loss: 3.966389608383179
0 (10 / 20), loss: 3.220976543426514
0 (15 / 20), loss: 3.115802764892578
Train Epoch 0, loss: 3.1995686531066894
1 (5 / 20), loss: 3.415890598297119
1 (10 / 20), loss: 2.586612892150879
1 (15 / 20), loss: 2.966762399673462
Train Epoch 1, loss: 2.802999997138977
2 (5 / 20), loss: 3.252368927001953
2 (10 / 20), loss: 2.8847944259643556
2 (15 / 20), loss: 2.597514343261719
Train Epoch 2, loss: 2.6773049533367157
3 (5 / 20), loss: 3.323799419403076
3 (10 / 20), loss: 2.492830586433411
3 (15 / 20), loss: 2.492376613616943
Train Epoch 3, loss: 2.5782290399074554
4 (5 / 20), loss: 2.930303144454956
4 (10 / 20), loss: 2.680714416503906
4 (15 / 20), loss: 2.2272167444229125
Train Epoch 4, loss: 2.486887913942337
5 (5 / 20), loss: 2.686834716796875
5 (10 / 20), loss: 2.620230865478516
5 (15 / 20), loss: 2.4685410976409914
Train Epoch 5, loss: 2.413221871852875
```
Now increase the number of layers:
```
$>python encoder_decoder.py --num_layers 3
0 (5 / 20), loss: 3.9147329807281492
0 (10 / 20), loss: 3.1582162857055662
0 (15 / 20), loss: 3.013378858566284
Train Epoch 0, loss: 3.1156901478767396
1 (5 / 20), loss: 3.3008444786071776
1 (10 / 20), loss: 2.6119103908538817
1 (15 / 20), loss: 2.8822034358978272
Train Epoch 1, loss: 2.763860309123993
2 (5 / 20), loss: 3.122182035446167
2 (10 / 20), loss: 2.8247500896453857
2 (15 / 20), loss: 2.6102036476135253
Train Epoch 2, loss: 2.6452258229255676
3 (5 / 20), loss: 3.2155590057373047
3 (10 / 20), loss: 2.54751296043396
3 (15 / 20), loss: 2.442613697052002
Train Epoch 3, loss: 2.5268495798110964
4 (5 / 20), loss: 2.6969152688980103
4 (10 / 20), loss: 2.6239708185195925
4 (15 / 20), loss: 2.220409655570984
Train Epoch 4, loss: 2.4248587071895598
5 (5 / 20), loss: 2.6048757314682005
5 (10 / 20), loss: 2.482684588432312
5 (15 / 20), loss: 2.298362302780151
Train Epoch 5, loss: 2.345057821273804
```

### Encoder-Decoder Model with Attention:
Finally, you are ready to implement an encoder-decoder with attention.
Complete the scaffolding code in `src.models.EncoderDecoderAttention` class. 
For this part you ONLY need to fill in the `__init__` and `forward` methods.
Run the commands below.
The serialized model is shown in the log below for additional help.
```
$>python encoder_decoder.py --attn
EncoderDecoderAttention(
  (dropout): Dropout(p = 0.0, axes=())
  (encoder): RNN(100 -> 100, TNC, bidirectional)
  (src_embedding): Embedding(29 -> 100, float32)
  (attention): Dense(300 -> 1, linear)
  (decoder_cell): SequentialRNNCell(
  (0): RNNCell(300 -> 100, tanh)
  )
  (output): Dense(100 -> 27, linear)
  (tgt_embedding): Embedding(27 -> 100, float32)
)
0 (5 / 20), loss: 4.020750331878662
0 (10 / 20), loss: 3.190213918685913
0 (15 / 20), loss: 3.011557197570801
Train Epoch 0, loss: 3.1529087901115416
1 (5 / 20), loss: 3.359003448486328
1 (10 / 20), loss: 2.6640984058380126
1 (15 / 20), loss: 2.9189178943634033
Train Epoch 1, loss: 2.8004651188850405
2 (5 / 20), loss: 3.1226752281188963
2 (10 / 20), loss: 2.8514935970306396
2 (15 / 20), loss: 2.706024932861328
Train Epoch 2, loss: 2.6716036796569824
3 (5 / 20), loss: 3.3150551319122314
3 (10 / 20), loss: 2.5321135997772215
3 (15 / 20), loss: 2.482138919830322
Train Epoch 3, loss: 2.572513198852539
4 (5 / 20), loss: 2.9278846263885496
4 (10 / 20), loss: 2.578867769241333
4 (15 / 20), loss: 2.2435038566589354
Train Epoch 4, loss: 2.4736402869224547
5 (5 / 20), loss: 2.702746772766113
5 (10 / 20), loss: 2.4243874073028566
5 (15 / 20), loss: 2.4484590530395507
Train Epoch 5, loss: 2.378611516952515
```

Again, your implementation should support increasing the number of layers, run the command below:
```
$>python encoder_decoder.py --attn --num_layers 3
EncoderDecoderAttention(
  (dropout): Dropout(p = 0.0, axes=())
  (encoder): RNN(100 -> 100, TNC, num_layers=3, bidirectional)
  (src_embedding): Embedding(29 -> 100, float32)
  (attention): Dense(300 -> 1, linear)
  (decoder_cell): SequentialRNNCell(
  (0): RNNCell(300 -> 100, tanh)
  (1): RNNCell(None -> 100, tanh)
  (2): RNNCell(None -> 100, tanh)
  )
  (output): Dense(100 -> 27, linear)
  (tgt_embedding): Embedding(27 -> 100, float32)
)
0 (5 / 20), loss: 3.9412882804870604
0 (10 / 20), loss: 3.129477596282959
0 (15 / 20), loss: 2.7870242595672607
Train Epoch 0, loss: 3.0631630659103393
1 (5 / 20), loss: 3.2382267475128175
1 (10 / 20), loss: 2.510899782180786
1 (15 / 20), loss: 2.967649745941162
Train Epoch 1, loss: 2.7748427510261537
2 (5 / 20), loss: 2.9372633934020995
2 (10 / 20), loss: 2.968257713317871
2 (15 / 20), loss: 2.712240982055664
Train Epoch 2, loss: 2.648495090007782
3 (5 / 20), loss: 3.226962375640869
3 (10 / 20), loss: 2.541613531112671
3 (15 / 20), loss: 2.3700276613235474
Train Epoch 3, loss: 2.509793370962143
4 (5 / 20), loss: 2.7343871116638185
4 (10 / 20), loss: 2.5602133750915526
4 (15 / 20), loss: 2.217679572105408
Train Epoch 4, loss: 2.422837036848068
5 (5 / 20), loss: 2.6835421085357667
5 (10 / 20), loss: 2.3960949659347532
5 (15 / 20), loss: 2.3987390995025635
Train Epoch 5, loss: 2.359060651063919
```

### Inference:
Where are the translations? So far you have implemented just the training methods. To obtain translations you need to fill in the TODOs in the `inference` method in `src.model.EncoderDecoderAttention`.
Enable inference in the output log with the following command:
```
python encoder_decoder.py --attn --inference
EncoderDecoderAttention(
  (dropout): Dropout(p = 0.0, axes=())
  (encoder): RNN(100 -> 100, TNC, bidirectional)
  (src_embedding): Embedding(29 -> 100, float32)
  (attention): Dense(300 -> 1, linear)
  (decoder_cell): SequentialRNNCell(
  (0): RNNCell(300 -> 100, tanh)
  )
  (output): Dense(100 -> 27, linear)
  (tgt_embedding): Embedding(27 -> 100, float32)
)
0 (5 / 20), loss: 4.020750331878662
0 (10 / 20), loss: 3.190214014053345
0 (15 / 20), loss: 3.0115572929382326
Train Epoch 0, loss: 3.1529088497161863
c r i s i s-->e e s s s
r e s p e c t o-->e e e
r e n u n c i a-->e e e
a l e g a n-->e e e
m u j e r e s-->e e e
d i g i t a l e s-->e e e
t e n e m o s-->e e e
e s t a d i o-->e e e
s u s-->e e e
j ó v e n e s-->e e e
d e t e n i d o-->e e e
e s t o s-->e e e
t i e m p o-->e e e
a ñ o s-->e e e
p r o y e c t o s-->e e e
c o n t i n ú a-->e e e
c o n f l i c t o-->e e e
t a m b i é n-->e e e
a q u í-->e e e
d i v i d i d a-->e e e
```

After 100 epochs this was my model's predictions:
```
99 (5 / 20), loss: 0.004960099584423006
99 (10 / 20), loss: 0.0020660983864217997
99 (15 / 20), loss: 0.002092824154533446
Train Epoch 99, loss: 0.002972277856315486
c r i s i s-->c r i s e s
r e s p e c t o-->r e s p e i t s
r e n u n c i a-->r e n ú n c i a
a l e g a n-->a l e g a m
m u j e r e s-->m u l h e r e s
d i g i t a l e s-->d i g i t i i s
t e n e m o s-->t e m o s
e s t a d i o-->e s t á d s
s u s-->s e u s
j ó v e n e s-->j o v e s
d e t e n i d o-->d e t i d s
e s t o s-->e s t s e s
t i e m p o-->t e m p o
a ñ o s-->a n o s
p r o y e c t o s-->p r o j e i o s
c o n t i n ú a-->c o n t i o u
c o n f l i c t o-->c o n f l i t o
t a m b i é n-->t a m b
a q u í-->a q u i
d i v i d i d a-->d i v i d i
```

#### Credits
This assignment was created by [Adi Renduchintala](https://arendu.github.io/) with assitance from [Shuyang Ding](http://www.cs.jhu.edu/~sding/) for the [JSALT Summer School 2018](https://www.clsp.jhu.edu/workshops/18-workshop/2018-jhu-summer-school-on-human-language-technology/) Neural Machine Translation tutorial. Contact [Adi Renduchintala](https://arendu.github.io/) for solutions (completed code).
