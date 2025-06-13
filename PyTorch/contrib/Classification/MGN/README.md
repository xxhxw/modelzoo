# MGN
MGN（Multiple Granularity Network）是一个多分支的深度网络，采用了将全局信息和各粒度局部信息结合的端到端特征学习策略。
## 1. 模型概述
- [![MGN](https://arxiv.org/abs/1804.01438v1)](https://arxiv.org/abs/1804.01438v1)
- 仓库链接：[MGN](https://github.com/GNAYUOHZ/ReID-MGN.git)
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

用户自行获取 `Market-1501` 数据集，将数据集上传到服务器任意路径下并解压。您可以点击[此链接](https://zheng-lab-anu.github.io/Project/project_reid.html)从公开网站中下载数据集。


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
    cd <ModelZoo_path>/PyTorch/contrib/Classification/ReID-MGN
    ```
2. 运行训练。该模型支持单机单卡。
    ```
    export TORCH_SDAA_AUTOLOAD=cuda_migrate  #自动迁移环境变量
    python main.py --mode train --data_path path_to/Market-1501-v15.09.15 
    ```
### 2.5 训练结果
输出训练loss曲线及结果（参考使用[get_loss.py](./get_loss.py)）: 

Mean Relative Error (MRE): 0.001747
Mean Absolute Error (MAE): 0.474734

Test Result:
PASS - MRE (0.001747) <= 0.05 or MAE (0.474734) <= 0.0002
