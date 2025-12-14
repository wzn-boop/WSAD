# DeepTSAD


install `DeepOD`
```shell
git clone https://github.com/xuhongzuo/DeepOD
pip install .
```
spa的使用？能否修改成pa类似的api，输入是ground-truth label和分数，输出是调整后的分数
- LAT网络结构
- 测试脚本
- 先搜我们的参数，再跑对比



```shell
Time: 04-21 08.52.07, n_known: 10, two_steps: False 
Data: ASD, Algo: rosas, Runs: 1
Parameters,	 [data_type], 			  ts
Parameters,	 [prt_steps], 			  5
Parameters,	 [seq_len], 			  30
Parameters,	 [stride], 			  1
Parameters,	 [batch_size], 			  32
Parameters,	 [epochs], 			  30
Parameters,	 [epoch_steps], 			  40
Parameters,	 [lr], 			  8e-05
Parameters,	 [rep_dim], 			  256
Parameters,	 [network], 			  Transformer
Parameters,	 [act], 			  GELU
Parameters,	 [hidden_dims], 			  512
Parameters,	 [d_model], 			  64
Parameters,	 [pos_encoding], 			  fixed
Parameters,	 [attn], 			  cc_attn
Parameters,	 [alpha], 			  0.5
Parameters,	 [margin], 			  5
Parameters,	 [eta], 			  0.5
Parameters,	 [rep_regularization], 			  triplet
Parameters,	 [reg_loss], 			  smooth
```
