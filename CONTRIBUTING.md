# 介绍

各位亲爱的太初生态伙伴们！首先，我们衷心感谢您对我们太初的关注与支持。我们专注于为深度学习领域提供高性能，高能效的智能算力解决方案，目前已经成功适配了业界广泛认可的两个主要框架——PyTorch和PaddlePaddle。我们相信，通过与您的紧密合作和共同探索，我们的硬件产品将能够更好地服务于AI社区，推动人工智能技术的发展与应用。

热烈欢迎各位伙伴一起来参与我们的开源项目，共同推进人工智能技术的发展。

# 贡献要求
开发者提交的模型包括适配后的运行代码、LICENSE、README、Dockerfile、run_scripts等， 并遵循以下标准。

## ModelZoo License规则
1. 在所有完全自主开发的代码文件和头文件内容最上方补充如下内容（提供C/C++、Python两种版本，请根据代码语言进行选择）：

C/C++ License
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
Python License
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

2. 对已有第三方版权声明的原文件基础上进行修改的，以文件为单位增加注释说明，为减轻工作量，建议统一注释为`// Adapted to tecorigin hardware`或`# Adapted to tecorigin hardware`。


 ## README

README向用户介绍模型的功能、使用方法、性能、精度、数据集、环境依赖等信息，README文件应包含如下内容：

- 简介：

    1. 模型的介绍，包含模型的出处和算法介绍。
    
    2. 数据集准备:数据集获取方法，数据集处理方法。

    2. Docker环境准备的方法: 包括获取SDAA基础Docker环境，创建当前模型Docker镜像，创建当前模型Docker容器，启动当前模型Docker容器等。

    3. 启动训练的方法: 包括进入对应启动脚本的目录，和启动的方法。

    4. 训练结果: 使用表格提供简单的参数介绍，和对应的精度和性能结果。

- 关键要求：
    1. 模型的精度应当达到原始模型水平。

    2. 模型的训练过程应当使用DDP(Distributed Data Parallel)和AMP(Automatic Mixed Precision)。


README写作可参考如下链接：

[ResNet50 TecoPaddle README](https://gitee.com/tecorigin/modelzoo/tree/main/PaddlePaddle/Classification/ResNet)

 ## 统一接口
为方便模型的使用和测试，Tecorigin ModelZoo提供统一的模型运行接口和日志接口。
### 统一运行接口

统一使用run_scirpts接口规则进行模型运行，
该目录下至少应当包括:

1. README: 参数介绍至少应当有model_name，batchsize，epoch或step，nnode，node_rank参数。

2. run_script.py: 使用argparse对参数进行解析，并转换为str格式的启动脚本，并使用os.system()启动脚本。
请参考: [ResNet50 TecoPaddle run_scirpts](https://gitee.com/tecorigin/modelzoo/tree/main/PaddlePaddle/Classification/ResNet/run_scripts)。

### 统一日志接口

统一使用tcap_dlloger输出统一的log日志，用于分析模型运行过程和结果.
使用方法请参考:[tcap_dlloger README](https://gitee.com/xiwei777/tcap_dlloger)。

## Dockerfile
本仓库所有模型都基于Docker环境进行部署，Docker镜像环境的准备使用Dockerfile文件进行配置。
Dockerfile文件可以参考: [TecoPaddle ResNet50 Dockerfile](https://gitee.com/tecorigin/modelzoo/blob/main/PaddlePaddle/Classification/ResNet/Dockerfile)。


## 路径规范
贡献者提交的模型路径应当为:<框架名>/contrib/<算法领域>/<模型名称>。
1. 框架名当前包括PyTorch或PaddlePaddle。
2. 算法领域当前有Classification、Detection、Face、GNN、NLP、Recommendation、Reinforcement、Segmentation、Speech，请开发者从上述中选择。如果所选模型不在上述列表中，可使用其他算法领域名称，并在issue中对此进行说明。
3. 模型名称即是对应的模型名称。

例如GoogleNet的PyTorch版本提交的路径为为: PyTorch/contrib/Classification/GoogleNet。

## PR(Pull Requests)提交
1. 请fork Tecorigin/ModelZoo仓库至开发者账号下，基于开发者账号下的ModelZoo进行工作。完成开发内容后提交Pull Requests，源分支选择开发分支，目标分支选择tecorigin/modelzoo:main。

    建议工作分支名称命名为contrib/<开发者团队名称>/<模型名称>，例如contrib/jiangnan_university_ailab/deeplabv3。

2. PR标题：请在PR标题前标注活动名称，开发者团队名称及适配的内容。

    例如参与[【生态活动】元碁智汇·定义未来](https://gitee.com/tecorigin/teco-torch/issues/I9HG17?from=project-issue)时，标题请参考 **【生态活动】元碁智汇·定义未来-江南大学AILAB-在PyTorch框架上支持resnet50在imagenet上的训练**

3. PR内容：PR内容应当包括如下具体信息：
    - 本次提交代commit id链接：应当给到具体的commit id，当有新的feature commit后，开发者应当编辑此处，更新至最新的commit id。
    - 当前适配的软件栈版本：在python中import torch_sdaa/paddle_sdaa即可打印当前软件栈版本。
    - 源码参考：应当给出具体的参考链接和对应的commit id或tag。
    - 工作目录：请参考路径规范。
    - 适配内容及对应的运行脚本：提测脚本应当使用run_script的方式运行。
    - 结果展示：结果展示用应当包含数据集，模型精度结果及模型运行脚本运行时间。
    - README自测结果：确定README已经通过自测，非开发者可以通过README运行此次PR内容。

具体PR提测内容可以参考模板：[【生态活动】元碁智汇·定义未来-江南大学AILAB-在PyTorch框架上支持resnet50在imagenet上的训练【请勿合入，仅作为PR模板进行展示】](https://gitee.com/tecorigin/modelzoo/pulls/10)

## 编程规范
- Python代码遵循[PEP8](https://peps.python.org/pep-0008/)规范。

## commit信息提交建议
- commit message建议使用[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)规范。