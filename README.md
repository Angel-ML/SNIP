# SNIP

## 1.项目介绍

 SNIP 为一款集成了“精调+蒸馏+模型加速”一体的模型上线助力工具。用户只需要提供经过 post-train 以后的预训练模型和业务精调数据，并且修改 3~8 行训练代码， SNIP就会自动输出加速后的模型。

通过对多种量化与蒸馏算法的深度优化，我们相比实现了更高的加速效果，同时在多业务上都成功适配了行之有效的无损加速手段。

当前初步在AI、视频、文章相关场景取得了不错的效果。

工具包基于python3.8开发，支持作为一个python包进行使用，同时本身作为一个工具包，对环境的依赖程度低。

注意：目前Auto48X只支持调用自己的modeling文件（目前只支持BertForSequenceClassification，如果需要扩展支持请联系负责人）。用户需要自己intialize： 

model: original model that is used for quantization or distillation, this is the student model w.r.t. knowledge distillation.

optimizer (for model): optimizer for optimizating the model.

teacher model: the teacher model. In pure_distillation, this is the larger model. In model quantization, this is the original uncompressed model or a larger model.

optimizer_param_group (the parameter_group that is used for initialize the parameter group)。

## 2.功能特性

- 轻量级

  工具包作为Python 包，执行简单安装后，通过简单导入即可使用。

- 易使用

  用户只需要修改七行用户脚本代码即可训练量化后的模型

- 适用于神经网络量化与蒸馏训练场景

## 3.快速上手

### Build SNIP：

目前Auto48X支持pip instal 使用（setuptools==50.3.2）以及直接import项目文件夹使用两种方式。

```
git clone /Auto48X.git 
cd Auto48X 
pip install .
```

##### **requires：**

python >= 3.8

transformers==4.19.0

torch better be 1.11.0

\-------------------------------------------------------------------------------------------------------------------

### EXAMPLE

1) 标准AUTO48 qat with distillation。

  见 Auto48/example/ 路径下脚本。

### BUILD STEP 

1）配置Auto48X所需参数。

Auto48X默认参数配置使用 Auto48X/config/auto48_default.json 文件配置。用户如需自定义Auto48超参，可通过args和json文件配置参数，**args优先级高于json文件**。

修改parser用来接收Auto48X所需参数（分别放在解析parser之前和之后）：

```
parser = Auto48X.add_core_argument(parser) #用来添加Auto48需要用的arguments args = parser.parse_args() args = Auto48X.config_auto48(args) #设置一些Auto48的默认设置
若不打算使用argparse来初始化Auto48X参数，可以使用如下代码，使用dict修改Auto48X参数。
auto48x_args = Auto48X.set_auto48_args(
{            
"ddp": True,            
"qat": True,            
"local_rank": args.local_rank,            
"auto48_config": args.auto48x_config,        
})
```

2）初始化Teacher model：

```
model = modeling.BertForSequenceClassification(config=bert_config) model = model.to(device) teacher_model = modeling.BertForSequenceClassification(config=bert_config) teacher_model.to(device) #初始化teacher model checkpoint = torch.load(os.path.join(args.model_path, "pytorch_model.bin"), map_location="cpu") teacher_model.load_state_dict(checkpoint, strict= True) #load teacher model的参数 model.load_state_dict(checkpoint, strict = True)
```



3）初始化Auto48引擎：

```
model, optimizer, teacher_model, engine = Auto48X.Auto48Init(args, model, optimizer=optimizer,optimizer_param_group=optimizer_grouped_parameters,teacher_model=teacher_model, lr_scheduler=scheduler)
```



4）运行forward：

```
outputs = engine.engine_forward(batch)
```



注意这里forward不能直接model(batch)运行，需要用engine.engine_forward(batch)来运行。返回的第一个值是正常的BERT的返回类。（这个只需要在training的时候调用，evaluate的时候正常model(batch)就好了）



5）添加蒸馏：



```
num_input = torch.sum(attention_mask, dim=1) loss = engine.add_knowledge_distillation(loss, num_input=num_input) 
```



注意这里num_input是每个sample的token数，可以直接用



6）进行backward并更新模型：



```
engine.backward(loss) engine.step()
```



\-------------------------------------------------------------------------------------------------------------------

*如果是从老版本Auto48X迁移过来的用户，一共有三处需要修改的地方：

1）outputs, _, _ = engine.engine_forward(batch) 修改成 outputs = engine.engine_forward(batch)。

2）model.backward(loss)修改成engine.backward(loss)。

3）model.step()修改成engine.step()。

4）若直接使用model，model的输出是一个dict数据

\-------------------------------------------------------------------------------------------------------------------

### SNIP启动教程：

#### Deepspeed :

```
deepspeed user_script.py \ --distillation_loss_scale 1 \ --distillation_attention_scale 1 \ --lr_input 0.01 \ --lr_weight 0.005 \ --qat user args commands etc...
Auto48X默认依赖于DeepSpeed（为了加速训练并且方便调用fp16和bp16）。
参照https://www.deepspeed.ai/getting-started/#launching-deepspeed-training里面的launching部分。
```

#### Python3 :

使用DDP :

```
python3 -m torch.distributed.launch --nproc_per_node 1 user_script.py \ --distillation_loss_scale 1 \ --distillation_attention_scale 1 \ --lr_input 0.01 \ --lr_weight 0.005 \ --qat --ddp user args commands etc...
```

不使用DDP：

```
python3 user_script.py \
--distillation_loss_scale 1 \
--distillation_attention_scale 1 \ 
--lr_input 0.01 \ 
--lr_weight 0.005 \ 
--qat  \
--disable_deepspeed user args commands etc...
```

\-------------------------------------------------------------------------------------------------------------------

## 4.常见问题

Auto48X参数指引（具体定义可以参照auto48_utils.py）：

注意事项：

  （1）--qat、--pure_distillation、–pure_qat_eval、–pure_qat_eval 至少需要指定一项。

  （2）命令台输入的args优先级高于指定的json文件。

  （3）--pure_qat_eval模式需要模型先过Auto48Init后再load_from_ckpt。

  （4）使用fp16训练需要安装apex包。



**参数列表：**

##### Mode

| Auto48_mode       | Help                              |
| ----------------- | --------------------------------- |
| qat               | 启用qat模式。                     |
| pure_finetune     | 无QAT纯finetune模式。             |
| pure_distillation | 无QAT纯蒸馏模式。                 |
| pure_qat_eval     | 支持返回纯qat模型以供eval的模式。 |

##### All hyper-parameter

| Name                         | Help                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| auto48_config                | Auto48X的参数config路径，默认config已经在Auto48X/config/auto48_default.json中 |
| qat                          | 启用qat模式。                                                |
| do_calib                     | 用来设置calibration的flag。                                  |
| calib_step                   | calibration的次数，一般两次就好。                            |
| teacher                      | 用来load teacher model的路径。                               |
| distillation_loss_scale      | 用来控制蒸馏强度，一般设置在10。                             |
| distillation_attention_scale | 用来设置attention蒸馏的强度，一般设置1~10。                  |
| distillation_encode_scale    | 设置为1即可。                                                |
| lr_input                     | QAT里面input quantizer的学习率，一般0.05即可。不可以过大。   |
| lr_weight                    | QAT里面weight quantizer的学习率，一般0.01即可。不可以过大。  |
| int4_layers                  | 设置需要做int4量化的层（1_2代表第1,2层做int4量化，从第0层开始）。如果是None，则默认做全int8量化。 |
| annealing_T                  | 退火蒸馏参数，默认是1。                                      |
| pure_finetune                | 无QAT纯finetune模式。                                        |
| pure_distillation            | 纯蒸馏模式。                                                 |
| pure_qat_eval                | 支持返回纯qat模型以供eval的模式。                            |
| KD_function                  | 蒸馏算法。目前支持minilm,minilmV2,multilayer蒸馏。建议minilm。 |
| model_fp32                   | 使用fp32来运行model。                                        |
| teacher_fp32                 | 使用fp32来运行teacher model。                                |
| bp16                         | 使用bp16来运行model。                                        |
| disable_deepspeed            | 不使用deepspeed的模式。                                      |
| ddp                          | 使用pytorch DDP做并行计算。需要和--disable_deepspeed配合使用。 |
| load_from_calib              | 禁用自动加载校准模型的标志                                   |
| add_pooled_outputs_kd        | 添加pool output蒸馏算法                                      |

**--pure_qat_eval模式**

1）配置Auto48所需参数

```
parser = Auto48.add_core_argument(parser) #用来添加Auto48需要用的arguments args = parser.parse_args() args = Auto48.config_auto48(args) #设置一些Auto48的默认设置
```



2）生成模型

```
import Auto48X.tools.auto48_qat_eval as auto48_qat_eval model_qat_eval = auto48_qat_eval.Auto48_qat_eval(args, model) ... #use your model ...
```



蒸馏细节：

对于不同的用户脚本，支持数据集无label蒸馏和有label蒸馏。

Auto48X 中 engine.add_knowledge_distillation(loss, num_input) 内置了对student和teacher的output进行loss计算。 可以通过Input中loss设置为None时代表用户的loss无法从数据集label获得。

