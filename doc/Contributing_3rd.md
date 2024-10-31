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
3. 在代码中添加必要的注释和License。
4. 确保sdaa上精度存在下降趋势。
5. 在script目录下添加运行脚本。
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

### 3.3 在代码中添加必要的注释和License
对于代码中重要的部分，需要加入注释介绍功能，帮助用户快速熟悉代码结构，包括但不仅限于：
- 函数的功能说明。
- init，save，load，等io部分。
- 运行中间的关键状态，如print，save model等。

在所有完全自主开发的代码文件和头文件内容最上方补充如下内容：
对于C/C++文件需要添加 C/C++ License:
```
// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.
```

对于Python文件需要添加 Python License：
```
# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
```

对已有第三方版权声明的原文件基础上进行修改的，以文件为单位增加注释说明，为减轻工作量，建议统一注释为
```
// Adapted to tecorigin hardware 或
# Adapted to tecorigin hardware。
```

### 3.4 确保sdaa上精度存在下降趋势
需要保证模型的loss或metric在sdaa上存在下降趋势。如果训练时长大于2小时，则以2小时为训练时间；如果训练时长小于2小时，则完成完整的训练。如果适配完，发现模型精度存在问题，可以参考精度调测文档[PyTorch](http://docs.tecorigin.com/release/tecopytorch/)和 [PaddlePaddle](http://docs.tecorigin.com/release/tecopaddle)中的模型调测章节进行排查。
- 精度结果和精度曲线图应该包含loss和metric
- 如果metric数据数量小于5，则只需要保证loss具备下降趋势
- 如果在cuda上2小时候训练，loss和metric均无明显下降趋势，请在PR和README中提供对应的说明

### 3.5 在script目录下添加运行脚本
在指定目录下添加启动脚本：
1. 添加process_data.sh 便于处理数据，如果数据不需要处理，则不需要提供
2.添加 train_sdaa_3rd.sh，便于启动训练，复现结果
3. 添加训练结果的日志train_sdaa_3rd.log
4. 添加plot_curve.py，便于对loss和metric进行可视化
5. 添加精度曲线图train_sdaa_3rd.png

### 3.6 添加README文档
README向用户介绍模型的功能、使用方法、精度、数据集、环境依赖等信息，README文件应包含如下内容：

- 简介：

    1. 模型的介绍：包含模型的出处和算法介绍。

    2. 数据集准备：数据集获取方法，数据集处理方法。

    3. Docker环境准备的方法: 包括获取SDAA基础Docker环境，创建当前模型Docker镜像，创建当前模型Docker容器，启动当前模型Docker容器等。

    4. 启动训练的方法: 包括进入对应启动脚本的目录，和启动的方法。

    5. 训练结果: 提供参数介绍，精度结果和精度曲线图。

- 关键要求：
    1. 模型的精度应当达到原始模型水平。

    2. 模型的训练过程中，使用DDP(Distributed Data Parallel)和AMP(Automatic Mixed Precision)来提升性能。

    3. 如果使用开源数据集或权重，提供开源获取方式和数据处理方法。如果使用非开源数据集或权重，请提供百度网盘下载链接和数据处理方法。

    4. 如果当前开发环境无法使用Docker，请详细描述环境搭建过程（TecoTorch/TecoPaddle及更基础的依赖可忽略）。
    
    5. 请确保requirements.txt正确无误，可以通过pip install -r requirements.txt安装。


README写作可参考如下链接：

[ResNet50 TecoTorch README](https://gitee.com/tecorigin/modelzoo/tree/main/PyTorch/Classification/ResNet)

### 3.7 PR提交
完成上述所有流程后，参考[PR提交规范](./PullRequests_3rd.md)提交代码。
