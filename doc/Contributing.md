# 模型训练适配指南
## 1. 前期准备
适配之前需要确认源码、模型和数据集，具体流程如下：
1. 确认适配的源码，源码优先级：指定源码>官方源码>第三方开源>个人开源。
2. 确认适配的模型具体信息，例如resnet有多个系列，应明确是哪一个版本，例如：resnet50。
3. 确认需要适配的数据集和对应的metric指标。
4. 在PyTorch/PaddlePaddle的cpu或gpu环境复现源码提供的metric指标，确保源码、模型和数据集的准确无误。

## 2. 训练环境准备
### 2.1 docker容器环境
当前提供的环境已经包含所有的基础环境及训练框架(PyTorch,PaddlePaddle)环境，用户根据算法需求安装x86其他依赖即可。
可以在使用前根据[FAQ](https://gitee.com/tecorigin/modelzoo/issues/I9RVL5?from=project-issue)中的1和2自查当前环境是否可用。

```
(base) root@DevGen03:/softwares# conda info -e
# conda environments:
#
base                  *  /root/miniconda3
paddle_env               /root/miniconda3/envs/paddle_env
torch_env                /root/miniconda3/envs/torch_env
tvm-build                /root/miniconda3/envs/tvm-build

(base) root@DevGen03:/softwares# conda activate torch_env
(torch_env) root@DevGen03:/softwares# python -c "import torch,torch_sdaa"
--------------+------------------------------------------------
 Host IP      | N/A
 PyTorch      | 2.0.0a0+gitdfe6533
 Torch-SDAA   | 1.6.0b0+git19f8ed9
--------------+------------------------------------------------
 SDAA Driver  | 1.1.2b1 (N/A)
 SDAA Runtime | 1.1.2b0 (/opt/tecoai/lib64/libsdaart.so)
 SDPTI        | 1.1.0 (/opt/tecoai/lib64/libsdpti.so)
 TecoDNN      | 1.19.0b3 (/opt/tecoai/lib64/libtecodnn.so)
 TecoBLAS     | 1.19.0b3 (/opt/tecoai/lib64/libtecoblas.so)
 CustomDNN    | 1.19.0b1 (/opt/tecoai/lib64/libtecodnn_ext.so)
 TecoRAND     | 1.6.0b0 (/opt/tecoai/lib64/libtecorand.so)
 TCCL         | 1.16.0b0 (/opt/tecoai/lib64/libtccl.so)
--------------+------------------------------------------------
(torch_env) root@DevGen03:/softwares# teco-smi
Wed Jun  5 02:46:48 2024
+-----------------------------------------------------------------------------+
|  TCML: 1.10.0        SDAADriver: 1.1.2b1        SDAARuntime: 1.1.2b0        |
|-------------------------------+----------------------+----------------------|
| Index  Name                   | Bus-Id               | Health      Volatile |
|        Temp          Pwr Usage|          Memory-Usage|             SPE-Util |
|=============================================================================|
|   0    TECO_AICARD_01         | 00000000:01:00.0     | OK                   |
|        35C                90W |        0MB / 15296MB |                   0% |
|-------------------------------+----------------------+----------------------|
|   1    TECO_AICARD_01         | 00000000:01:00.0     | OK                   |
|        35C                90W |      165MB / 15296MB |                   0% |
|-------------------------------+----------------------+----------------------|
|   2    TECO_AICARD_01         | 00000000:01:00.0     | OK                   |
|        35C                90W |      165MB / 15296MB |                   0% |
|-------------------------------+----------------------+----------------------|
|   3    TECO_AICARD_01         | 00000000:01:00.0     | OK                   |
|        35C                90W |      165MB / 15296MB |                   0% |
+-------------------------------+----------------------+----------------------+
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  Tcaicard     PID      Process name                            Memory Usage |
|=============================================================================|
|     1       76262      python3.8                                     165 MB |
|     2       76263      python3.8                                     165 MB |
|     3       76264      python3.8                                     165 MB |
+-----------------------------------------------------------------------------+
```

## 3. 适配流程
### 3.1 整体流程
1. 根据第一章节内容完成前期准备，确保源码、模型和数据集的准确性，对齐精度指标。
2. 将PyTorch或Paddle代码迁移至sdaa。
3. 添加统一运行接口。
4. 添加统一日志接口。
5. sdaa上训练精度对齐cuda。
6. 根据README样板，添加README文档。
7. 根据PR提交规范提交代码。

### 3.2 代码迁移至sdaa
#### 3.2.1 device适配
确认好源码仓库后，需要将源码的运行device适配sdaa。这里举一个简单的例子，基于源码[参考链接](https://gitee.com/xiwei777/modelzoo/commit/7afd230a59658d5a78e9c626bce09eaaf54f7519)，进行sdaa device的迁移。具体[迁移代码修改](https://gitee.com/xiwei777/modelzoo/commit/7e693fd7bb4f78e796451f8bcb214a9786b163f8)，可点击对应链接查看。

也可在太初文档官网查看模型迁移文档：[PyTorch](http://docs.tecorigin.com/release/tecopytorch/)和 [PaddlePaddle](http://docs.tecorigin.com/release/tecopaddle)中的模型迁移章节。

#### 3.2.2 DDP适配
sdaa设备具有单卡4SPA特性，因此需要适配分布式训练DDP，具体适配手册参考[PyTorch](http://docs.tecorigin.com/release/tecopytorch/)和 [PaddlePaddle](http://docs.tecorigin.com/release/tecopaddle)中的模型训练章节中的分布式训练DDP文档。

#### 3.2.3 AMP适配
为加快训练速度，需要适配自动混合精度AMP训练，具体适配手册参考[PyTorch](http://docs.tecorigin.com/release/tecopytorch/)和 [PaddlePaddle](http://docs.tecorigin.com/release/tecopaddle)中的模型迁移章节中的自动混合精度AMP文档。

### 3.3 添加统一运行接口
我们希望所有的模型都可以用一种启动的规则都可以运行起来，用户/测试/或者其他任何人不需要去理解算法业务启动脚本的逻辑。这样会大大降低后续其他开发者使用的成本。因此 Tecorigin ModelZoo提供统一的模型运行接口。

统一使用`run_scirpts`接口规则进行模型运行，
该目录下至少应当包括:

1. `README.md`: 参数介绍至少应当有model_name，batch_size，epoch，step，dataset_path，lr，device，autocast等参数。针对单机DDP测例，额外添加nproc_per_node，针对多机DDP测例，额外添加nnode，node_rank参数。

2. `argument.py`：抽象出所有参数进行单独管理。

3. `formate_cmd.py`：执行指令的格式化输出。

4. `run_modelname.py`: 其中`modelname`为具体的模型名，如`run_resnet.py`。该脚本使用argparse对参数进行解析，并转换为str格式的启动脚本，同时进行格式化，再使用os.system()启动指令。请参考[Bert](../PyTorch/NLP/BERT/run_scripts/run_bert_base_imdb.py)的书写格式组织启动命令。

注意：统一运行接口需要支持--step参数，用来短训测试功能。

统一运行接口添加具体请参考: [Bert TecoPyTorch run_scirpts](../PyTorch/NLP/BERT/run_scripts)。

### 3.4 添加统一日志接口
统一使用`tcap_dllogger`输出统一的log日志，用于分析模型运行过程中的指标和运行结果。至少在训练阶段需要ips和loss，在验证阶段需要metric。如果你模型使用的并非ips来衡量性能，也可以采用其他指标。

tcap_dllogger工具使用方法请参考:[README](https://gitee.com/xiwei777/tcap_dllogger)。

具体修改[参考代码](https://gitee.com/xiwei777/modelzoo/commit/e8dabf0d3b1cebc2c13ecd1a125c9946ec2c342e)，大体分三步：

第一步：在训练之前初始化logger
```
# 初始化logger
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity
json_logger = Logger(
    [
        StdOutBackend(Verbosity.DEFAULT),
        JSONStreamBackend(Verbosity.VERBOSE, 'dlloger_example.json'),
    ]
)
json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
```
第二步：获取需要输出的一些数值，比如，loss，ips等信息
第三步：输出日志
```
json_logger.log(
step = (epoch, step),
data = {
        "rank":os.environ["LOCAL_RANK"],
        "train.loss":loss,
        "train.ips":ips,
        },
verbosity=Verbosity.DEFAULT,
)
```

记得把git+https://gitee.com/xiwei777/tcap_dllogger.git添加到requirements.txt。

### 3.5 sdaa上精度对齐cuda
该部分需要保证适配完sdaa后，使用相同参数，训练精度和cuda对齐，相对误差不能大于1%。如果适配完，发现模型精度存在问题，可以参考精度调测文档[PyTorch](http://docs.tecorigin.com/release/tecopytorch/)和 [PaddlePaddle](http://docs.tecorigin.com/release/tecopaddle)中的模型调测章节进行排查。

### 3.6 添加README文档
README向用户介绍模型的功能、使用方法、精度、数据集、环境依赖等信息，README文件应包含如下内容：

- 简介：

    1. 模型的介绍：包含模型的出处和算法介绍。

    2. 数据集准备：数据集获取方法，数据集处理方法。

    3. Docker环境准备的方法: 包括获取SDAA基础Docker环境，创建当前模型Docker镜像，创建当前模型Docker容器，启动当前模型Docker容器等。

    4. 启动训练的方法: 包括进入对应启动脚本的目录，和启动的方法。

    5. 训练结果: 使用表格提供简单的参数介绍，和对应的精度结果。

- 关键要求：
    1. 模型的精度应当达到原始模型水平。

    2. 模型的训练过程中，使用DDP(Distributed Data Parallel)和AMP(Automatic Mixed Precision)来提升性能。

    3. 如果使用开源数据集或权重，提供开源获取方式和数据处理方法。如果使用非开源数据集或权重，请提供百度网盘下载链接和数据处理方法。


README写作可参考如下链接：

[ResNet50 TecoPaddle README](https://gitee.com/tecorigin/modelzoo/tree/main/PaddlePaddle/Classification/ResNet)

### 3.7 添加模型的yaml信息
用户在[model.yaml](../PyTorch/contrib/model_config/model.yaml)中补充相关的参数设置，用于PR的功能性测试。功能性测试包含两部分检测：

- 目录结构规范性检测：检查提交的模型目录下是否包含`run_scripts`，`README.md`，`requirements.txt`等必要文件以及目录。目录结构如下：

        └── model_dir
            ├──requirements.txt
            ├──README.md
            ├──run_scripts
               ├──run_modelname.py
               ├──argument.py
               ├──formate_cmd.py
               ├──REAMDE.md

- 模型训练功能性检查：检查提交的指令是否正常跑通，没有功能性错误。

yaml文件的具体信息参考[model yaml](../PyTorch/contrib/model_config/README.md)。

### 3.8 PR提交
完成上述所有流程后，参考[PR提交规范](./PullRequests.md)提交代码。
