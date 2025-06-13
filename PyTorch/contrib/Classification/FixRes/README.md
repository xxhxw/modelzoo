# FixRes
FixRes是一个图像分类网络，该模型使用较低分辨率图像输入对ResNet50网络进行训练，并使用较高分辨率图像输入对训练好的模型进行finetune，最终使用较高分辨率进行测试，以此解决预处理过程中图像增强方法不同引入的偏差。
## 1. 模型概述
- [![FixRes](https://arxiv.org/abs/1906.06423)](https://arxiv.org/abs/1906.06423)
- 仓库链接：[FixRes](https://github.com/facebookresearch/FixRes)
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

用户自行获取 `ImageNet2012` 数据集，将数据集上传到服务器任意路径下并解压。


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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/FixRes
    ```
2. 运行训练。该模型支持单机单卡。
    ```
    export TORCH_SDAA_AUTOLOAD=cuda_migrate  #自动迁移环境变量
    python main_resnet50_scratch.py --batch 64 --num-tasks 1 --learning-rate 2e-2 
    ```
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[get_loss.py](./get_loss.py)）: 

Mean Relative Error (MRE): -0.037431
Mean Absolute Error (MAE): 0.230986

Test Result:
PASS - MRE (-0.037431) <= 0.05 or MAE (0.230986) <= 0.0002

