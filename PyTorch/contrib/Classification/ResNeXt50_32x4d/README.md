# ResNeXt50_32x4d

## 1.模型概述

ResNeXt50_32x4d是一种高效的卷积神经网络模型，由Facebook AI Research (FAIR) 团队在2017年提出，属于ResNeXt系列的一个代表性架构。ResNeXt是在经典的ResNet（残差网络）基础上改进而来，通过引入“分组卷积”（Grouped Convolutions）和“基数”（Cardinality）的概念，进一步提升了模型的性能，同时保持了较低的计算复杂度。 

## 2.快速开始

使用本模型执行训练的主要流程如下：

1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

1. 虚拟环境启动：

    ```bash
    cd <ModelZoo_path>/PyTorch/contrib/Classification/ResNeXt50_32x4d
    
    conda activate torch_env
    
    # 执行以下命令验证环境是否正确，正确则会打印版本信息
    python -c 'import torch_sdaa'
    ```

2. 安装python依赖：

    ```bash
    pip install -r requirements.txt
    ```

### 2.2 准备数据集

ResNeXt50_32x4d运行在ImageNet数据集上，数据集配置可以参考[https://blog.csdn.net/xzxg001/article/details/142465729](https://gitee.com/link?target=https%3A%2F%2Fblog.csdn.net%2Fxzxg001%2Farticle%2Fdetails%2F142465729) 

### 2.3 启动训练

该模型支持单机单核组、单机单卡 

**单机单核组**

```
python train.py --dataset_path /data/datasets/imagenet --batch_size 64 --epochs 20 --distributed False --lr 0.01 --autocast True
```

**单机单卡**

```
torchrun --nproc_per_node=4 train.py --dataset_path /data/datasets/imagenet --batch_size 64 --epochs 20 --distributed True --lr 0.01 --autocast True
```

### 2.4 训练结果

| 加速卡数量 |      模型       | 混合精度 | Batch Size | Iteration | train_loss | AccTop1 |
| :--------: | :-------------: | :------: | :--------: | :-------: | :--------: | :-----: |
|     1      | ResNeXt50_32x4d |   amp    |     64     |    51     |     /      |    /    |

#### 情况说明：

在实际训练过程中，在2h的时间内，一个epoch中只能运行51个iteration，训练速度慢。
