# Experiment



Here are the corresponding GLUE scores on the test set:

|           | MNLI          | cola    | mrpc   | rte        | stsb   | sst2   | qqp    | qnli    |
| --------- |---------------| ------- |--------|------------| ------ | ------ | ------ | ------- |
| Base-ckpt | 0.8466-0.8420 | 0.5338  | 0.9135 | 0.6750     | 0.8763 | 0.9243 | 0.8779 | 0.9105  |
| INT8      | 0.8446/0.8401 | 0.58147 | 0.9170 | 0.7039 | 0.8745 | 0.9266 | 0.8778 | 0.91177 |

F1 scores are reported for QQP and MRPC, Spearman correlations are reported for STS-B, and accuracy scores are reported for the other tasks.

 For each task, we selected the best fine-tuning learning rate (among  3e-5, and 2e-5) , bactchsize among 12,96 ,on the Dev set, distillation_loss_scale among 10 100 200,distillation_attention_scale among 0 1。





Here are the corresponding SQUAD scores on the test set:

|      | EM | F1 |
| ---- | ---- | ---- |
| BERT   | 80.9 | 88.3 |
| INT8   | 80.7 | 88.1 |

We selected the best fine-tuning learning rate 1e-5,distillation_loss_scale  200,distillation_attention_scale 0。