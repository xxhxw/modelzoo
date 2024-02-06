# BERT finetune
 
BERT（Bidirectional Encoder Representations from Transformers）是一种预训练的语言模型，它在大规模的无监督数据上进行了训练，学习到了丰富的语言表征，拥有强大的自然语言理解能力（Natural Language Understanding, NLU）。对于NLP子任务，可使用标注数据集在BERT预训练模型的基础上进行有监督训练，使BERT模型进一步理解任务，从而对子任务有更好的效果。

## 目录
<!-- toc -->
- [环境搭建](#1-环境搭建)
    - [代码拉取](#11-代码拉取)
    - [预训练权重](#12-预训练权重)
    - [版本控制](#13-版本控制)
        - [docker环境](#131-docker环境)
        - [物理机环境](#132-物理机环境)
- [NLP任务支持](#2-nlp任务支持)
    - [问答任务-使用SQuAD v1.1数据集](#21-问答任务-使用squad-v11数据集)
        - [数据集获取](#211-数据集获取)
        - [训练及验证](#212-训练及验证)
        - [训练结果](#213-训练结果)
    - [文本摘要任务-使用CNN/DM数据集](#22-文本摘要任务-使用cnndm数据集)
        - [数据集获取](#221-数据集获取)
        - [rouge安装](#222-rouge安装)
        - [训练及验证](#223-训练及验证)
        - [训练结果](#224-训练结果)
    - [情感分析任务-使用IMDb数据集](#23-情感分析任务-使用imdb数据集)
        - [数据集获取](#231-数据集获取)
        - [训练及验证](#232-训练及验证)
        - [训练结果](#233-训练结果)
    - [文本分类-使用THUCNews数据集](#24-文本分类-使用thucnews数据集)
        - [数据集获取](#241-数据集获取)
        - [训练及验证](#242-训练及验证)
        - [训练结果](#243-训练结果)
    - [通用参数说明](#25-通用参数说明)
<!-- toc -->
## 1 环境搭建

### 1.1 代码拉取

``` bash
git clone http://10.10.30.109/tecoap/modelzoo.git
```

### 1.2 预训练权重

本项目使用Nvidia提供的BERT-base-uncased预训练权重进行训练。可前往[NVIDIA-NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/models/bert_pyt_ckpt_base_pretraining_amp_lamb/files)下载。

中文BERT权重可前往[ModelScope](https://www.modelscope.cn/models/dienstag/chinese-bert-wwm/files)下载。

### 1.3 版本控制

#### 1.3.1 docker环境
- 进入Dockerfile所在目录，运行以下命令

``` bash
cd <modelzoo-root>/PyTorch/NLP/BERT
DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t  torch_bert_base
```

镜像内部软件栈版本信息如下:
[SDAA软件栈版本信息](../../../.dependencies.json)
|软件名|pytorch|sdaadriver|sdaart|tccl|dnn|blas|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|版本信息| 0.15.0 |0.15.0|0.16.0|1.11.0|1.12.0| 1.12.0|


创建BERT-base PyTorch sdaa docker容器：
``` bash
docker run  -itd --name bert_base_pt -v <path to dataset>:/workspace/dataset -v <path to checkpoint>:/workspace/checkpoint --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g torch_bert_base /bin/bash
```

- 参数介绍详见[Docker configuration](./docs/Docker_configuration.md)

进入docker容器
```
docker exec -it bert_base_pt /bin/bash
cd /workspace/NLP/BERT/run_scripts
conda activate torch_env
```

#### 1.3.2 物理机环境
``` bash
conda activate torchenv
cd <modelzoo-root>/PyTorch/NLP/BERT
pip install -r requirements.txt
```

另外由于计算文本生成指标rouge时依赖libxml-parser-perl，需要自行安装：

``` bash
sudo apt-get install libxml-parser-perl
```

## 2 NLP任务支持
本仓库对问答、文本摘要、情感分析、文本分类四个任务进行支持。

### 2.1 问答任务-使用SQuAD v1.1数据集

#### 2.1.1 数据集获取

``` bash
cd <modelzoo-root>/PyTorch/NLP/BERT/data/squad
# 下载完成后，会保存至<modelzoo-root>/PyTorch/NLP/BERT/data/squad/v1.1
sh squad_download.sh
```

#### 2.1.2 训练及验证

该任务支持单卡，单机八卡运行。

- Demo正确性测试
``` bash
cd <modelzoo-root>/PyTorch/NLP/BERT
# 注意修改dataset_path和checkpoint_path，分别指向数据集目录和预训练权重文件
python run_scripts/run_bert_base_squad_v1.1.py --model_name bert_base_uncased --nproc_per_node 4 --bs 4 --lr 3e-5 --device sdaa  --epoch 3 --step 10 --dataset_path path/to/dataset --grad_scale True --autocast True --checkpoint_path path/to/bert_base.pt --warm_up 0.1 --max_seq_length 384
```

- 单机单卡（4核）训练

``` bash
cd <modelzoo-root>/PyTorch/NLP/BERT
# 注意修改dataset_path和checkpoint_path，分别指向数据集目录和预训练权重文件
python run_scripts/run_bert_base_squad_v1.1.py --model_name bert_base_uncased --nproc_per_node 4 --bs 4 --lr 3e-5 --device sdaa --epoch 3 --dataset_path path/to/dataset --grad_scale True --autocast True --checkpoint_path path/to/bert_base.pt --warm_up 0.1 --max_seq_length 384 --do_predict --do_eval
```

- 单机八卡（8*4核）训练

``` bash
cd <modelzoo-root>/PyTorch/NLP/BERT
# 注意修改dataset_path和checkpoint_path，分别指向数据集目录和预训练权重文件
python run_scripts/run_bert_base_squad_v1.1.py --model_name bert_base_uncased --nproc_per_node 32 --bs 2 --lr 6e-5 --device sdaa --epoch 3 --dataset_path path/to/dataset --grad_scale True --autocast True --checkpoint_path path/to/bert_base.pt --warm_up 0.1 --max_seq_length 384 --do_predict --do_eval
```

##### 参数说明

见[通用参数说明](#25-通用参数说明)

支持--do_predict与--do_eval参数

#### 2.1.3 训练结果

| 芯片 |卡 |软件栈版本 |频率 | Epochs | 混合精度 |Batch size|max seq len| 吞吐量| Acc| extra_match|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1|teco 0.12.0|2.5G| 3|是|16|384| - | 89.01% | 82.10 |
|SDAA|8|teco 0.12.0|2.5G| 3|是|64|384| - | 88.61% | 81.47 |

<!-- |V100|1|cuda 11.7 | - |3 |是(apex)|4|384| - | 88.68% | - |
|V100|8|cuda 11.7 | - |3 |是(apex)|32|384| - | 87.42% | - | -->

### 2.2 文本摘要任务-使用CNN/DM数据集

#### 2.2.1 数据集获取

本任务的数据集来自[BertSum项目仓库](https://github.com/nlpyang/BertSum)，仓库中提供了[处理后的数据](https://drive.google.com/open?id=1x0d61LP9UAN389YN00z0Pv-7jQgirVg6)，可直接进行训练。


#### 2.2.2 rouge安装

为了计算文本生成指标rouge，需要进行如下操作：

``` bash
git clone https://github.com/andersjo/pyrouge.git rouge

# 注意这里需要配置绝对路径
pyrouge_set_rouge_path path/to/rouge/tools/ROUGE-1.5.5/

cd rouge/tools/ROUGE-1.5.5/data
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db

# 若该命令执行成功则表示rouge配置成功，后续可正常计算rouge指标
python -m pyrouge.test
```

#### 2.2.3 训练及验证

请务必按照2.2.2节中的流程完成rouge依赖的安装，否则测试阶段会出现报错

该任务支持单卡运行。

- Demo正确性测试
``` bash
cd <modelzoo-root>/PyTorch/NLP/BERT
# 注意修改dataset_path和checkpoint_path，分别指向数据集目录和预训练权重文件
python run_scripts/run_bert_base_cnndm.py --model_name bert_base_uncased --nproc_per_node 1 --bs 4 --lr 2e-3 --device sdaa --step 10 --dataset_path path/to/dataset --grad_scale True --autocast True --checkpoint_path path/to/bert_base.pt --warm_up 0.2 --max_seq_length 512
```

- 单机单卡（4核）训练

``` bash
cd <modelzoo-root>/PyTorch/NLP/BERT
# 注意修改dataset_path和checkpoint_path，分别指向数据集目录和预训练权重文件
python run_scripts/run_bert_base_cnndm.py --model_name bert_base_uncased --nproc_per_node 4 --bs 4 --lr 2e-3 --device sdaa --step 20000 --dataset_path path/to/dataset --grad_scale True --autocast True --checkpoint_path path/to/bert_base.pt --warm_up 0.2 --max_seq_length 512 --do_predict
```

##### 参数说明

见[通用参数说明](#25-通用参数说明)

CNN/DM数据集仅支持使用--step参数控制训练数据量

仅支持--do_predict参数，不支持--do_eval参数

#### 2.2.4 训练结果

| 芯片 |卡 |软件栈版本 |频率 | steps | 混合精度 |Batch size|max seq len| 吞吐量| ROUGE-1| ROUGE-2|ROUGE-L|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1|teco 0.12.0|2.5G| 20000 |是|16|512| - | 42.94 | 20.07 |39.40|


<!-- |CUDA|1|cuda 11.7|-| 20000 | 否 |16|512| - | 43.14 | 20.23 |39.59| -->

### 2.3 情感分析任务-使用IMDb数据集

#### 2.3.1 数据集获取

IMDb数据集是一个二分类情感任务的数据集，提供了25000个训练样本和25000个测试样本。

ACL官方提供了[原始数据集](https://ai.stanford.edu/~amaas/data/sentiment/)下载。

``` bash
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

# 下载完成后进行解压
mkdir imdb_dataset && tar -xvf aclImdb_v1.tar -C imdb_dataset

cd <modelzoo-root>/PyTorch/NLP/BERT/data
# 修改imdb_dir与output_dir，imdb_dir为数据集解压路径，output_dir为处理完后的数据保存路径
# 传入参数<path/to/imdb_dataset>与<path/to/processed_imdb>，<path/to/imdb_dataset>为数据集解压路径，<path/to/processed_imdb>为处理完后的数据保存路径
python process_imdb.py <path/to/imdb_dataset> <path/to/processed_imdb>

# 执行完成后processed_imdb中会生成train.tsv dev.tsv两个文件
```

#### 2.3.2 训练及验证

该任务支持单卡运行。

- Demo正确性测试
``` bash
cd <modelzoo-root>/PyTorch/NLP/BERT
# 注意修改dataset_path和checkpoint_path，分别指向数据集目录和预训练权重文件
python run_scripts/run_bert_base_imdb.py --model_name bert-large-uncased --nproc_per_node 1 --bs 16 --lr 2.4e-5 --device sdaa --step 10 --dataset_path path/to/IMDB --checkpoint_path path/to/bert_base.pt --max_seq_length 128 --warm_up 0.1 --grad_scale True --autocast True
```

- 单机单卡（4核）训练

``` bash
cd <modelzoo-root>/PyTorch/NLP/BERT
# 注意修改dataset_path和checkpoint_path，分别指向数据集目录和预训练权重文件
python run_scripts/run_bert_base_imdb.py --model_name bert-large-uncased --nproc_per_node 4 --bs 16 --lr 2.4e-5 --device sdaa --epoch 4 --dataset_path path/to/IMDB --checkpoint_path path/to/bert_base.pt --max_seq_length 128 --warm_up 0.1 --grad_scale True --autocast True --do_eval
```

##### 参数说明

见[通用参数说明](#25-通用参数说明)

仅支持--do_eval参数，不支持--do_predict参数

#### 2.3.3 训练结果

| 芯片 |卡 |软件栈版本 |频率 | Epochs | 混合精度 |Batch size|max seq len| 吞吐量|acc|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1|teco 0.12.0|2.5G| 4 |是|64|128| - | 84.42% |

<!-- |A100|1|cuda 11.7|-| 4 |是|16|128| - | 84.26% | -->

### 2.4 文本分类-使用THUCNews数据集

#### 2.4.1 数据集获取

THUCNews是一个多分类的中文新闻数据集，包括 财经, 彩票, 房产, 股票, 家居, 教育, 科技, 社会, 时尚, 时政, 体育, 星座, 游戏, 娱乐 14种新闻。默认使用混合精度训练，可以用两种方法获取训练使用的数据集。

清华官方提供了[THUCNews数据集](http://thuctc.thunlp.org/)下载。

``` bash

# 下载完成后进行解压
mkdir thucnews_dataset && unzip THUCNews.zip -d thucnews_dataset

cd <modelzoo-root>/PyTorch/NLP/BERT/data
# 传入参数<path/to/thucnews_dataset>与<path/to/output_dir>，<path/to/thucnews_dataset>为数据集解压路径，<path/to/processed_thucnews>为处理完后的数据保存路径
python process_thucnews.py <path/to/thucnews_dataset> <path/to/processed_thucnews> 

# 执行完成后processed_thucnews中会生成train.tsv dev.tsv两个文件
```

#### 2.4.2 训练及验证

该任务支持单卡运行。

- Demo正确性测试
``` bash
cd <modelzoo-root>/PyTorch/NLP/BERT
# 注意修改dataset_path和checkpoint_path，分别指向数据集目录和预训练权重文件
python run_scripts/run_bert_base_thucnews.py --model_name bert-base-chinese --nproc_per_node 1 --bs 16 --lr 2.4e-5 --device sdaa --step 10 --dataset_path path/to/processed_thucnews --checkpoint_path path/to/pytorch_model.bin --max_seq_length 128 --warm_up 0.1 --grad_scale True --autocast True
```

- 单机单卡（4核）训练
``` bash
cd <modelzoo-root>/PyTorch/NLP/BERT
# 注意修改dataset_path和checkpoint_path，分别指向数据集目录和预训练权重文件
python run_scripts/run_bert_base_thucnews.py --model_name bert-base-chinese --nproc_per_node 4 --bs 16 --lr 2.4e-5 --device sdaa --epoch 4 --dataset_path path/to/processed_thucnews --checkpoint_path path/to/pytorch_model.bin --max_seq_length 128 --warm_up 0.1 --grad_scale True --autocast True --do_eval
```

##### 参数说明

见[通用参数说明](#25-通用参数说明)

仅支持--do_eval参数，不支持--do_predict参数

#### 2.4.3 训练结果

| 芯片 |卡 |软件栈版本 |频率 | Epochs | 混合精度 |Batch size|max seq len| 吞吐量|acc|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1|teco 0.12.0|2.5G| 4 |是|64|128| - | 95.43% |

<!-- |A100|1|cuda 11.7|-| 4 |是|16|128| - | 95.71% | -->

### 2.5 通用参数说明

参数名 | 解释 | 样例
-----------------|-----------------|-----------------
model_name |模型名称 | --model_name bert_base_uncased
epoch| 训练轮次，和训练步数冲突 | --epoch 3
step | 训练步数，和训练轮数冲突 | --step 10
batch_size/bs | 每个rank的batch_size | --batch_size 4 / --bs 4
dataset_path | 数据集路径 | --dataset_path path/to/dataset
nproc_per_node | DDP时，每个node上的rank数量。不输入时，默认为1，跑单核 | --nproc_per_node 4
lr|学习率|--lr 3e-5
device|设备类型|--device cuda/--device sdaa
autocast|开启amp autocast|--autocast True
grad_scaler| 使用grad_scale | --grad_scale True
checkpoint_path| 预训练权重路径 | --checkpoint_path path/to/bert_base.pt
warm_up| warm_up比例 | --warm_up 0.1
max_seq_length|输入的最大句子长度| --max_seq_length 384
do_predict|训练时验证|--do_predict
do_eval|测试集测试|--do_eval