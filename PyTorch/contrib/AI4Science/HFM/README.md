
# 基于HFM的CFD

## 1. 模型概述
PINN（Physics-Informed Neural Networks）是一类结合了物理方程和神经网络的机器学习模型，主要用于求解偏微分方程（PDEs）等物理问题。HFM（Hybrid Physics-Based and Data-Driven Models）则是一种将基于物理的模型和数据驱动的模型结合起来的方法。在HFM中，PINN可以发挥重要作用，通过融合物理定律和数据来改进模型的精度和泛化能力。

注：此项目代码来自于太初的同事

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何使用创建模型训练时所需的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考[基础环境安装](../../../doc/Environment.md)章节，完成训练前的基础环境检查和安装。

### 2.2 准备数据集
#### 2.2.1 获取数据集

此项目运行的数据集为随机生成的模拟数据集，运行cfd/Datasets_HFM/gen_test_data.py文件即可生成。无需下载数据集。
```
python gen_test_data.py
```
保存在cfd/Datasets_HFM/gen_data_predict_t1.npy


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
    此环境内已包含所有依赖包，如还需安装可运行。
    ```
    pip install -r requirements.txt
    ```

### 2.4 启动训练
1. 在构建好的环境中，进入训练脚本所在目录。
    ```
    cd PyTorch/contrib/AI4Science/HFM
    ```

2. 运行训练。
    - 单机单卡
    ```
        python run_scripts/run_script.py --device sdaa --T_data 10 --N_data 100 --total_epoch 5
    ```
    - 单机多卡
    ```
        python run_scripts/run_script.py --ddp --device sdaa --T_data 10 --N_data 100 --total_epoch 5 --local_size 4
    ```

    更多训练参数参考[README](run_scripts/README.md)

### 2.5 训练结果

结果可视化
![c](Results/draw/gen_data_predict_t1_res_rey100_test_3d_c.png)

最终各参数的error
Error c: 2.894351e-03
Error u: 3.342468e-03
Error v: 2.238216e-03
Error w: 7.272329e-03
Error p: 2.041813e-04