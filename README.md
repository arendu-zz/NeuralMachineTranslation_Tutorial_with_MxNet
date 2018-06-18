# jsalt2019-nmt-lab

### MXNet Setup:
Install mxnet and all the required dependencies with the following commands:
```
$>git clone http://github.com/szha/gluon-crash-course -b jsalt
$>cd gluon-crash-course
$>conda env create -f env/environment.yml 
```
### Lab Setup:
Clone this repo:
```
$>git clone https://github.com/arendu/jsalt2018-nmt-lab-public.git
$>cd jsalt2018-nmt-lab-public
$>source activate jsalt_gluon_nlp
```
### RNN Language Model:
Once you complete filling in all the `TODOs` in `src.models.RNNLM` run the command below and check if you match this output log:
```
$>python lm.py
Train Epoch 0, loss: 3.2715672731399534
Train Epoch 1, loss: 2.9671921730041504
Train Epoch 2, loss: 2.8810736417770384
Train Epoch 3, loss: 2.7949750661849975
Train Epoch 4, loss: 2.7448513865470887
Train Epoch 5, loss: 2.708175873756409
Train Epoch 6, loss: 2.651815187931061
Train Epoch 7, loss: 2.6001593232154847
Train Epoch 8, loss: 2.5436817049980163
Train Epoch 9, loss: 2.480805438756943
Train Epoch 10, loss: 2.4066086947917937
```
Now try increasing the number of layers in your RNNLM:
```
$>python lm.py --num_layers 3
Train Epoch 0, loss: 3.163326895236969
Train Epoch 1, loss: 2.9031516671180726
Train Epoch 2, loss: 2.8718849301338194
Train Epoch 3, loss: 2.7825149416923525
Train Epoch 4, loss: 2.740653133392334
Train Epoch 5, loss: 2.699657416343689
Train Epoch 6, loss: 2.64787814617157
Train Epoch 7, loss: 2.587969148159027
Train Epoch 8, loss: 2.511477828025818
Train Epoch 9, loss: 2.451095974445343
Train Epoch 10, loss: 2.35848348736763
```
### Encoder-Decoder Model for translation:
Next you will implement a simple encoder-decoder without attention.
Complete the scaffolding code in `src.models.EncoderDecoder` and run the command below:
```
$>python encoder_decoder.py
0 (5 / 20), loss: 3.9483273029327393
0 (10 / 20), loss: 3.208091640472412
0 (15 / 20), loss: 3.095386266708374
Train Epoch 0, loss: 3.17526136636734
1 (5 / 20), loss: 3.4073556423187257
1 (10 / 20), loss: 2.6471239566802978
1 (15 / 20), loss: 2.9639118671417237
Train Epoch 1, loss: 2.82170147895813
2 (5 / 20), loss: 3.328487682342529
2 (10 / 20), loss: 2.948730802536011
2 (15 / 20), loss: 2.684615230560303
Train Epoch 2, loss: 2.754596674442291
3 (5 / 20), loss: 3.4415226936340333
3 (10 / 20), loss: 2.584901762008667
3 (15 / 20), loss: 2.5777024745941164
Train Epoch 3, loss: 2.669452965259552
4 (5 / 20), loss: 3.073537015914917
4 (10 / 20), loss: 2.8063485622406006
4 (15 / 20), loss: 2.329734683036804
Train Epoch 4, loss: 2.6050745785236358
5 (5 / 20), loss: 2.8794105052948
5 (10 / 20), loss: 2.7639767646789553
5 (15 / 20), loss: 2.5976045608520506
Train Epoch 5, loss: 2.557079529762268
```
Now increase the number of layers:
```
$>python encoder_decoder.py --num_layers 3
0 (5 / 20), loss: 3.9218214988708495
0 (10 / 20), loss: 3.180419683456421
0 (15 / 20), loss: 3.0194921493530273
Train Epoch 0, loss: 3.13070627450943
1 (5 / 20), loss: 3.383973217010498
1 (10 / 20), loss: 2.685383176803589
1 (15 / 20), loss: 2.880984592437744
Train Epoch 1, loss: 2.8040854811668394
2 (5 / 20), loss: 3.1793892860412596
2 (10 / 20), loss: 2.871953582763672
2 (15 / 20), loss: 2.6747466564178466
Train Epoch 2, loss: 2.6990251898765565
3 (5 / 20), loss: 3.2255905628204347
3 (10 / 20), loss: 2.6241029262542725
3 (15 / 20), loss: 2.4569656372070314
Train Epoch 3, loss: 2.5741503477096557
4 (5 / 20), loss: 2.8779995441436768
4 (10 / 20), loss: 2.6343345642089844
4 (15 / 20), loss: 2.2493020057678224
Train Epoch 4, loss: 2.504862344264984
5 (5 / 20), loss: 2.744069218635559
5 (10 / 20), loss: 2.563315749168396
5 (15 / 20), loss: 2.476594352722168
Train Epoch 5, loss: 2.4543148636817933
```

### Encoder-Decoder Model with Attention:
Finally, you are ready to implement an encoder-decoder with attention.
Complete the scaffolding code in `src.models.EncoderDecoderAttention` class. 
For this part you ONLY need to fill in the `__ini__` and `forward` methods.
Tun the command below.
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
0 (5 / 20), loss: 3.9554842948913573
0 (10 / 20), loss: 3.1437063694000242
0 (15 / 20), loss: 2.9759069442749024
Train Epoch 0, loss: 3.1135228872299194
1 (5 / 20), loss: 3.379365253448486
1 (10 / 20), loss: 2.6872398853302
1 (15 / 20), loss: 2.9530708312988283
Train Epoch 1, loss: 2.8238080739974976
2 (5 / 20), loss: 3.206978273391724
2 (10 / 20), loss: 2.9199341773986816
2 (15 / 20), loss: 2.767395353317261
Train Epoch 2, loss: 2.737170135974884
3 (5 / 20), loss: 3.4210769653320314
3 (10 / 20), loss: 2.621690273284912
3 (15 / 20), loss: 2.5839598178863525
Train Epoch 3, loss: 2.6706495881080627
4 (5 / 20), loss: 3.0854537963867186
4 (10 / 20), loss: 2.7246798992156984
4 (15 / 20), loss: 2.351749849319458
Train Epoch 4, loss: 2.599903738498688
5 (5 / 20), loss: 2.862346124649048
5 (10 / 20), loss: 2.6246942043304444
5 (15 / 20), loss: 2.640377998352051
Train Epoch 5, loss: 2.5478350639343263
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
0 (5 / 20), loss: 3.8929978370666505
0 (10 / 20), loss: 3.081466245651245
0 (15 / 20), loss: 2.8262250900268553
Train Epoch 0, loss: 3.0467529892921448
1 (5 / 20), loss: 3.2727338790893556
1 (10 / 20), loss: 2.5672556877136232
1 (15 / 20), loss: 2.9814534187316895
Train Epoch 1, loss: 2.8001124382019045
2 (5 / 20), loss: 3.054044723510742
2 (10 / 20), loss: 2.994654989242554
2 (15 / 20), loss: 2.7612091064453126
Train Epoch 2, loss: 2.705072486400604
3 (5 / 20), loss: 3.3236002922058105
3 (10 / 20), loss: 2.6024903774261476
3 (15 / 20), loss: 2.4555612564086915
Train Epoch 3, loss: 2.5969822883605955
4 (5 / 20), loss: 2.8675305366516115
4 (10 / 20), loss: 2.673336458206177
4 (15 / 20), loss: 2.3176249504089355
Train Epoch 4, loss: 2.522545802593231
5 (5 / 20), loss: 2.800323486328125
5 (10 / 20), loss: 2.5203219175338747
5 (15 / 20), loss: 2.52230224609375
Train Epoch 5, loss: 2.466957634687424
```

### Inference:
Where are the translations? So you you have implemented just the training methods. To obtain translations you need to fill in the TODOs in the `inference` method in `src.model.EncoderDecoderAttention`.
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
0 (5 / 20), loss: 3.9554842948913573
0 (10 / 20), loss: 3.1437063217163086
0 (15 / 20), loss: 2.975906991958618
Train Epoch 0, loss: 3.1135228991508486
c r i s i s-->e e e
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
99 (5 / 20), loss: 0.11365396194159985
99 (10 / 20), loss: 0.03295676745474339
99 (15 / 20), loss: 0.07633087364956737
Train Epoch 99, loss: 0.07905320851132273
c r i s i s-->c r i s e s
r e s p e c t o-->r e s p e i t s
r e n u n c i a-->r e n ú n c i a
a l e g a n-->a l e g a m
m u j e r e s-->m u l h e r e s
d i g i t a l e s-->d i g i t a u s
t e n e m o s-->t e m o s
e s t a d i o-->e s t á d o
s u s-->s e u s
j ó v e n e s-->j o o e e s
d e t e n i d o-->d e t i d o
e s t o s-->e s t e s
t i e m p o-->t e m p o
a ñ o s-->a n o s
p r o y e c t o s-->p o o j e t o o
c o n t i n ú a-->c o n t i n u a
c o n f l i c t o-->c o n f l i t o
t a m b i é n-->t a m b a m
a q u í-->a q u i
d i v i d i d a-->d i v i d i u
```

#### Credits
This assignment was created by [Adi Renduchintala](https://arendu.github.io/) with assitance from [Shuyang Ding](http://www.cs.jhu.edu/~sding/) for the [JSALT Summer School 2018](https://www.clsp.jhu.edu/workshops/18-workshop/2018-jhu-summer-school-on-human-language-technology/) Neural Machine Translation tutorial. Contact [Adi Renduchintala](https://arendu.github.io/) for solutions (completed code).
