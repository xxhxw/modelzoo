# PR 提交规范
本文档给出开发者在提交PR(Pull Requests)时的规范事项，以供参考。
## 路径规范
贡献者提交的模型路径应当为:`<框架名>/contrib/<算法领域>/<模型名称>`。

- 框架名当前包括PyTorch或PaddlePaddle。
- 算法领域请参考模型列表中的分类。
- 模型名称即是对应的模型名称。

例如GoogleNet的PyTorch版本提交的路径为: `PyTorch/contrib/Classification/GoogleNet`，`PaddlePaddle/contrib/AI4Science/allen_cahn`。

## 命名规范
- 文件和文件夹命名中，使用下划线"_"代表空格，不要使用"-"。
- 类似权重路径、数据集路径、shape等参数，需要通过合理传参统一管理。
- 文件和变量的名称定义过程中需要能够通过名字表明含义。
- 在代码中定义path时，需要使用os.path.join完成，禁止使用string拼接的方式。


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
    - 结果展示：结果展示用应当包含模型精度结果，精度曲线图及模型运行脚本运行时间。
    - README自测结果：确定README已经通过自测，非开发者可以通过README运行此次PR内容。

具体PR提测内容可以参考模板：[【生态活动】元碁智汇·定义未来-江南大学AILAB-模型训练-在PyTorch框架上支持resnet50在imagenet上的训练【请勿合入，仅作为PR模板进行展示】](https://gitee.com/tecorigin/modelzoo/pulls/10)

## 编程规范
- Python代码遵循[PEP8](https://peps.python.org/pep-0008/)规范。

## commit信息提交建议
- commit message建议使用[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)规范。