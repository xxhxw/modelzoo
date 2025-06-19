# Evo-Levit
Evo-Levit的具体框架设计，包括基于全局class attention的token选择以及慢速、快速双流token更新两个模块。其根据全局class attention的排序判断高信息token和低信息token，将低信息token整合为一个归纳token，和高信息token一起输入到原始多头注意力（Multi-head Self-Attention, MSA）模块以及前向传播（Fast Fed-forward Network, FFN）模块中进行精细更新。更新后的归纳token用来快速更新低信息token。全局class attention也在精细更新过程中进行同步更新变化。
## 1. 模型概述
- [Evo-ViT](https://arxiv.org/abs/2108.01390)
- 仓库链接：[Evo-ViT](https://github.com/YifanXu74/Evo-ViT)
- 其他配置参考README_en.md

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考基础环境安装章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集和预训练模型

- 用户自行获取 `ImageNet2012` 数据集，将数据集上传到服务器任意路径下并解压。
- `Evo-Vit` 模型训练需要配置 `teacher—model` ，用户自行获取 `regnety_160-a5fe301d.pth` 预训练模型，可参考GitHub的[Evo-Vit](https://github.com/YifanXu74/Evo-ViT)。将获取的预训练模型放置在源码包根目录下。与源码中的配置参数的默认值 `./regnety_160-a5fe301d.pth` 保持一致。


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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/Evo-Levit
    ```
2. 运行训练。该模型支持单机单卡。
    ```
    export TORCH_SDAA_AUTOLOAD=cuda_migrate

    torchrun --nproc_per_node=1  main_levit.py     --model EvoLeViT_256_384     --input-size 384     --batch-size 64     --data-path <path>/dataset/imagenet     --output_dir <path>/Evo-ViT/output_dir 
    ```
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[get_loss.py](./get_loss.py)）: 

Mean Relative Error (MRE): 0.001798
Mean Absolute Error (MAE): 0.018771

Test Result:
PASS - MRE (0.001798) <= 0.05 or MAE (0.018771) <= 0.0002

