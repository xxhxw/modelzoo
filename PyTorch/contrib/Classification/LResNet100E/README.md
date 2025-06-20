# LResNet100
LResNet100是一个目标识别网络模型，该网络提出一种新的损失函数Additive Angular Margin Loss（ArcFace），通过深度卷积神经网络DCNN学习的特征嵌入，可以有效地增强目标识别的判别能力。
## 1. 模型概述
- [LResNet100](https://arxiv.org/abs/1801.07698)(https://arxiv.org/abs/1801.07698)
- 仓库链接：[LResNet100](https://github.com/TreB1eN/InsightFace_Pytorch)
- 其他配置参考README_en.md

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集

用户自行获取 `faces_emore` 原始数据集，将数据集上传到服务器模型源码包根目录的 `data` 目录下并解压。可参考[源码仓](https://github.com/TreB1eN/InsightFace_Pytorch/blob/master/README.md)的方式获取数据集。
数据集目录结构参考如下所示。

   ```
   data
    |-- data_pipe.py
	|-- faces_emore
            |-- agedb_30
            |-- calfw
            |-- cfp_ff
            |-- cfp_fp
            |-- cfp_fp
            |-- cplfw
            |-- imgs
            |-- lfw
            |-- vgg2_fp   
    ```
    > **说明：** 
    >该数据集的训练过程脚本只作为一种参考示例。

### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip install -r requirements.txt

    ```
### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd <ModelZoo_path>/PyTorch/contrib/Classification/LResNet100E
    ```
2. 运行训练。该模型支持单机单卡。
    ```
    cd /${模型文件夹名称} 
    # 创建日志存储，模型存储目录
    rm -rf ./work_space/* 
    mkdir ./work_space/history && mkdir ./work_space/log && mkdir ./work_space/models && mkdir ./work_space/save

    export TORCH_SDAA_AUTOLOAD=cuda_migrate  #自动迁移环境变量
    python train.py
    ```
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[get_loss.py](./get_loss.py)）: 

Mean Relative Error (MRE): 0.009545
Mean Absolute Error (MAE): 0.397602

Test Result:
PASS - MRE (0.009545) <= 0.05 or MAE (0.397602) <= 0.0002

