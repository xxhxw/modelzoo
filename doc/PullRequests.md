# PR 提交规范
本文档给出开发者在提交PR(Pull Requests)时的规范事项，以供参考。
## 路径规范
贡献者提交的模型路径应当为:`<框架名>/contrib/<算法领域>/<模型名称>`。

- 框架名当前包括PyTorch或PaddlePaddle。
- 算法领域当前有Classification、Detection、Face、GNN、NLP、Recommendation、Reinforcement、Segmentation、Speech，请开发者从上述中选择。如果所选模型不在上述列表中，可使用其他算法领域名称，并在issue中对此进行说明。
- 模型名称即是对应的模型名称。

例如GoogleNet的PyTorch版本提交的路径为: `PyTorch/contrib/Classification/GoogleNet`。

## 命名规范
- 文件和文件夹命名中，使用下划线"_"代表空格，不要使用"-"。
- 类似权重路径、数据集路径、shape等参数，需要通过合理传参统一管理。
- 文件和变量的名称定义过程中需要能够通过名字表明含义。
- 在代码中定义path时，需要使用os.path.join完成，禁止使用string拼接的方式。

## 注释和License
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

## PR(Pull Requests)提交
- 请fork [Tecorigin/ModelZoo](https://gitee.com/tecorigin/modelzoo/tree/main)仓库至开发者账号下，基于开发者账号下的ModelZoo仓库进行工作。完成开发内容后提交Pull Requests，源分支选择开发分支，目标分支选择`tecorigin/modelzoo:main`。
建议工作分支名称命名为`contrib/<开发者团队名称>/<模型名称>`，例如`contrib/jiangnan_university_ailab/resnet`。

- PR标题：请在PR标题前标注活动名称，开发者团队名称及适配的内容。

    例如参与[【生态活动】元碁智汇·定义未来](https://gitee.com/tecorigin/modelzoo/issues/IAHGWN?from=project-issue)时，标题请参考 **【生态活动】元碁智汇·定义未来-江南大学AILAB-模型训练-在PyTorch框架上支持resnet50在imagenet上的训练**

- PR内容：PR内容应当包括如下具体信息：
    - 当前适配的软件栈版本：在python中import torch_sdaa/paddle_sdaa即可打印当前软件栈版本，以截图的方式提供即可。
    - 源码参考：应当给出具体的参考链接和对应的commit id或tag，如果无参考源码，请说明。
    - 工作目录：请参考路径规范。
    - 适配内容及对应的运行脚本：提测脚本应当使用run_script的方式运行。
    - 结果展示：结果展示用应当包含模型精度结果及模型运行脚本运行时间。
        -  如果为完整的训练或微调任务，请提供最终的metric结果。
        -  如果为短训，请提供loss曲线图和最终的loss结果。
    - README自测结果：确定README已经通过自测，非开发者可以通过README运行此次PR内容。

具体PR提测内容可以参考模板：[【生态活动】元碁智汇·定义未来-江南大学AILAB-模型训练-在PyTorch框架上支持resnet50在imagenet上的训练【请勿合入，仅作为PR模板进行展示】](https://gitee.com/tecorigin/modelzoo/pulls/10)

## 编程规范
- Python代码遵循[PEP8](https://peps.python.org/pep-0008/)规范。

## commit信息提交建议
- commit message建议使用[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)规范。