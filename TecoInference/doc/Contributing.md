# 模型推理适配指南
## 1. 前期准备
适配之前需要确认源码、模型和数据集，具体流程如下：
1. 确认适配的源码，源码优先级：指定源码>官方源码>第三方开源>个人开源。
2. 确认适配的模型具体信息，例如resnet有多个系列，应明确是哪一个版本和输入shape，例如：resnet50：input_shape:224x224，batch_size:1~128。
3. 确认需要适配的数据集和对应的metric指标。
4. 在PyTorch/PaddlePaddle的cpu或gpu环境复现源码提供的metric指标，确保源码、模型和数据集的准确无误。

## 2. 推理环境准备
### 2.1 docker容器环境
当前提供的环境已经包含所有的基础环境及推理框架(TecoinferenceEngine)、onnx等环境，用户根据算法需求安装x86其他依赖即可。

执行以下命令进行 TecoinferenceEngine 基础环境验证。
```
python -c "import tvm"

# 输出以下内容
---------------+---------------------------------------------
Host IP        | xx.xx.xx.xx
---------------+---------------------------------------------
Teco-infer     | 1.2.0rc0+git8b22872
TecoDNN        | 1.20.0
TecoBLAS       | 1.20.0
TecoCustom     | 1.20.0
TECOCC         | 1.11.0
SDAA Runtime   | 1.2.0
SDAA Driver    | 1.2.0
---------------+---------------------------------------------
```

以下onnx和TensortRT环境安装供参考。

### 2.2 ONNX环境（可选）
```
onnx>=1.12.0
onnxsim			# 用于简化模型
onnxruntime	# 用于测试onnx推理
onnxconverter_common	# 用于将模型权重转为float16格式

# for torch model
torch>=1.12.0

# for paddle model
paddlepaddle
paddle2onnx
```

### 2.3 TensortRT环境（可选）

1. 获取TensorRT压缩包
    在官网下载TensorRT-8.6.0.12.Linux.x86_64-gnu.cuda-11.8.tar.gz

2. 解压
    ```
    tar -xzvf TensorRT-8.6.0.12.Linux.x86_64-gnu.cuda-11.8.tar.gz
    ```
3. 添加环境变量
    ```
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:xxx/TensorRT-8.6.0.12/lib
    ```
    其中xxx需要替换为TensorRT-8.6.0.12的路径。

4. 安装
    ```
    cd TensorRT-8.6.0.12/python/
    python3 -m pip install tensorrt-8.6.0-cp38-none-linux_x86_64.whl
    ```
    注意: paddle模型测试TensorRT需要更换GPU版本的paddle。

## 3. 适配流程
### 3.1 整体流程
1. 根据第一章节内容完成前期准备，确保源码、模型和数据集的准确性，对齐评估指标。
2. 将PyTorch或Paddle权重转换为onnx格式文件。
3. 将源码中数据集加载、数据预处理、模型推理和后处理部分抽出，适配modelzoo推理接口和目录格式。
4. 使用适配好的代码，测试onnxruntime-cpu推理，对齐官方评估指标，劣化相对误差不超过0.5%。
5. 使用适配好的代码，测试TecoInferenceEngine推理，对齐onnxruntime-cpu评估指标，劣化相对误差不超过0.1%。
6. 根据README样板，添加README文档。
7. 根据PR提交规范提交代码。

### 3.2 导出onnx
注意：导出后的模型需要使用onnxsim进行简化，推荐动态shape导出。注意保存导出onnx的代码文件，后续需要提交到repo中。

#### 3.2.1 Pytorch示例

PyTorch模型权重转为onnx格式可参考如下代码：

```
import onnx
import onnxsim		# 用于简化模型
from onnxconverter_common import float16	# 用于将模型转为float16

import torch
import torchvision

# init model
resnet = torchvision.models.resnet50(pretrained=True)
resnet.eval()

# init dumpy_input
dumpy_input = torch.randn(1, 3, 224, 224)

# 静态shape导出
torch.onnx.export(resnet,
                  dummy_input,
                  "resnet.onnx",
                  opset_version=12,         	# ONNX opset版本
                  input_names=['input'],    	# 输入名称
                  output_names=['output'],  	# 输出名称
                  do_constant_folding=True, 	# 是否执行常量折叠优化
                  dynamic_axes=None,	# 是否使用动态shape，不使用默认为None
                 )

# 动态shape导出（推荐）
dynamic_dims = {'input': {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch'}}

torch.onnx.export(resnet,
                  dummy_input,
                  "resnet_dyn.onnx",
                  opset_version=12,         	# ONNX opset版本
                  input_names=['input'],    	# 输入名称
                  output_names=['output'],  	# 输出名称
                  do_constant_folding=True, 	# 是否执行常量折叠优化
                  dynamic_axes=dynamic_dims,	# 是否使用动态shape，不使用默认为None
                 )

# 以下动态静态均适用

# Checks
model_onnx = onnx.load("resnet_dyn.onnx")  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model

# Simplify
model_onnx, check = onnxsim.simplify(model_onnx)
assert check, 'assert check failed'

# convert_float_to_float16
model_onnx = float16.convert_float_to_float16(model_onnx)
onnx.save(model_onnx, "resnet_float16_dyn.onnx")
```

#### 3.2.2 Paddle示例
PyTorch模型权重转为onnx格式可参考如下代码：
```
import onnx
import onnxsim		# 用于简化模型
from onnxconverter_common import float16	# 用于将模型转为float16

import paddle
from paddle.vision.models import resnet50

# init model
model = resnet50(pretrained=True)

# 静态shape
input_spec = [
    paddle.static.InputSpec(
        shape=[1, 3, 224, 224], dtype="float32"),
]

# 动态shape
input_spec = [
    paddle.static.InputSpec(
        shape=[-1, 3, -1, -1], dtype="float32"),
]
paddle.onnx.export(model,
                   "resnet.onnx",
                   input_spec=input_spec,
                   opset_version=12)

# 以下动态静态均适用

# Checks
model_onnx = onnx.load("resnet.onnx")  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model

# Simplify
model_onnx, check = onnxsim.simplify(model_onnx)
assert check, 'assert check failed'

# convert_float_to_float16
model_onnx = float16.convert_float_to_float16(model_onnx)
onnx.save(model_onnx, "resnet_float16.onnx")
```

### 3.3 modelzoo代码适配
modelzoo小模型推理目录结构如下：

```
TecoInference/
├── contrib         # 注意，提交代码需要放在此目录内，其他目录下文件禁止修改。
│   ├── engine		# TecoinferenceEngine, 此处代码非必要禁止修改。
│   │   ├── base.py
│   │   ├── __init__.py
│   │   ├── tecoinfer_paddle.py     # paddle模型推理引擎模板
│   │   └── tecoinfer_pytorch.py    # torch模型推理引擎模板
│   ├── example		# 推理pipeline适配代码在该路径下，根据具算法方向选择文件夹或创建新文件夹
│   │   ├── classification
│   │   ├── detection
│   │	...
│   └── utils       # 数据集读取，预处理，后处理代码。
│       ├── datasets
│       │   └── __init__.py
│       ├── __init__.py
│       ├── postprocess
│       │   ├── __init__.py
│       │   ├── paddle
│       │   │   └── __init__.py
│       │   └── pytorch
│       │       └── __init__.py
│       └── preprocess
│           ├── __init__.py
│           ├── paddle
│           │   └── __init__.py
│           └── pytorch
│               └── __init__.py
├── doc
│   ├── Environment.md          # 环境说明
│   ├── Contributing.md         # 模型适配说明
│   ├── 
│   └── 
├── Dockerfile
├── engine
│   ├── base.py
│   ├── __init__.py
│   ├── tecoinfer_paddle.py
│   └── tecoinfer_pytorch.py
├── example
│   └── classification
│       └── resnet		# 示例模型
│           ├── example_multi_batch.py      # 文件夹推理
│           ├── example_single_batch.py     # 单样本推理
│           ├── example_valid.py            # 验证集推理评估
│           ├── images
│           │   ├── cat.png
│           │   └── hen.jpg
│           ├── README.md           # 模型使用说明
│           └── requirements.txt    # 依赖文件
├── README.md
├── teco-inference-model-benchmark  # 性能测试使用的pass文件，禁止修改。
├── tecoexec # 极限性能测试脚本
│   ├── README.md
│   ├── requirements.txt
│   ├── save_engine.py
│   ├── testcase_configs
│   │   └── tecoexec_config.yaml
│   └── test_tecoexec.py
└── utils # 数据集读取，预处理，后处理代码。
    ├── datasets # 数据集读取
    │   ├── image_classification_dataset.py
    │   └──  __init__.py
    ├── __init__.py
    ├── postprocess # 后处理
    │   ├── __init__.py
    │   ├── paddle
    │   │   └── __init__.py
    │   └── pytorch
    │       ├── classification.py
    │       └── __init__.py
    └── preprocess #
        ├── __init__.py
        ├── paddle
        │   └── __init__.py
        └── pytorch
            ├── classification.py
            └── __init__.py
```

#### 3.3.1 适配内容
参考[resnet](../example/classification/resnet/)与[utils](../utils/)，需要在`TecoInference/contrib/`目录下适配模型推理精度pipeline代码和对应的性能测试配置，具体如下:
```
# 目录：TecoInference/contrib/example/算法方向/算法名称
├── demos						# 推理demo样本数据
├── README.md					# README文档
├── example_multi_batch.py		# 文件夹推理脚本
├── example_single_batch.py		# 单个样本推理脚本
├── example_valid.py			# 数据集推理脚本
├── export_onnx.py					# onnx导出导出脚本
└── requirements.txt			# 依赖

# 目录：TecoInference/contrib/utils, pytorch/paddle根据算法源码选择
├──utils
├── datasets
│   └── 算法名称_dataset.py		# 数据集加载脚本
├── postprocess
│   ├── pytorch
│   │   └── 算法名称.py			# 后处理脚本
│		└── paddle
└── preprocess
    ├── pytorch
    │   └── classification.py	# 预处理脚本
    └── paddle
# 目录：TecoInference/contrib/tecoexec/tecoexec_config.yaml 极限性能测试配置文件
```

#### 3.3.2 推理pipeline适配说明

pipeline适配主要包括`example_valid.py`、`example_single_batch.py`和`example_multi_batch.py`三个推理了文件。

注意：TecoinferenceEngine在太初卡上运行单卡三、四SPA推理时，会在每个SPA上初始化一份模型进行推理，因此模型初始化的`batch_size=单卡推理的batch_size / 单卡SPA数量`。例如，当推理pipeline传入的onnx文件的batch_size是16，那么实际运行单卡四SPA推理时，实际传入的input的batch_size需要设置为64。

下面针对需要适配的三个文件进行说明：

`example_valid.py`：数据集推理代码，以下对关键内容进行说明，其余可以参考[resnet](../example/classification/resnet/example_valid.py)补充。

```
# 添加engine和utils路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

# 导入推理引擎TecoInferEngine、优化文件路径PASS_PATH
from engine.tecoinfer_pytorch import TecoInferEngine
from engine.base import PASS_PATH

# 导入数据加载器load_data、预处理preprocess和后处理postprocess模块
from utils.datasets.image_classification_dataset import load_data
from utils.preprocess.pytorch.classification import preprocess
from utils.postprocess.pytorch.classification import postprocess

# 获取单卡三/四SPA环境变量
MAX_ENGINE_NUMS = int(os.getenv('MAX_ENGINE_NUMS', 4))	# 三/四SPA环境变量

if __name__ == "__main__":
    # 动态shape的onnx文件需要指定运行时模型的输入shape, 按照模型输入设置
    input_size = [[max(batch_size // MAX_ENGINE_NUMS, 1), 3, shape, shape]] # 参考3.3.2的注意内容。
    
    # 初始化模型，支持onnx/tensorrt/tvm
    pipeline = TecoInferEngine(ckpt=ckpt,				# 模型的onnx文件路径
                               input_name=input_name,	# 导出模型onnx时的input_name
                               target=target,			# 可选:'onnx'进行onnxruntime-cpu推理、'sdaa'进行TecoInferenceEngine推理、'cuda'(paddle模型为'gpu')进行tensorRT推理
                               batch_size=batch_size,	# 数据集推理的batch_size
                               input_size=input_size,	# 指定初始化时的输入shape
                               dtype="float16", 		# 可选"float16"和"float32"，推荐"float16"
                               pass_path=pass_path,		# 推理框架优化文件，新适配模型设置为：PASS_PATH / "default_pass.py" 即可
                              )
    
    # load dataset
    val_loader = load_data(data_path, batch_size)
    
    # 统计性能
    e2e_time = []
    pre_time = []
    run_time = []
    post_time = []
    ips = []
    
    results = []
    # 遍历数据集进行推理，记录结果和性能数据
    for _, (input, target) in tqdm(val_loader):
        start_time = time.time()
        # 预处理, 需要将输入数据处理为np.ndarray或按照输入顺序处理为[np.ndarray, np.ndarray, ...]格式
        images = preprocess(input, dtype=opt.dtype)
        preprocess_time = time.time() - start_time
        
        # 进行推理，输出为numpy格式数据
        prec = pipeline(images)
        model_time = infer_engine.run_time
    
        # 后处理, 例如目标检测算法需要进行nms等
        result = postprocess(prec)
        infer_time = time.time() - start_time
    
        results.append(result)
    
        # 统计性能数据
        postprocess_time = infer_time - preprocess_time - model_time
        sps = batch_size / infer_time
        e2e_time.append(infer_time)
        pre_time.append(preprocess_time)
        run_time.append(pipeline.run_time)
        post_time.append(postprocess_time)
        ips.append(sps)
    
    # metric计算，根据算法方向计算数据集推理的评价指标
    metric = get_acc(results)

    # 释放device显存，stream等资源
    if "sdaa" in opt.target:
        infer_engine.release()
    
    # 打印结果
    print('eval_metric', metric)
    print(f'summary: avg_sps: {np.mean(ips)}, e2e_time: {sum(e2e_time)}, avg_inference_time: {np.mean(run_time)}, avg_preprocess_time: {np.mean(pre_time)}, avg_postprocess: {np.mean(post_time)}')
```

`example_single_batch.py`：单个样本推理(batch=1) ,以下对关键内容进行说明，其余可以参考[resnet](../example/classification/resnet/example_single_batch.py)补充。

```
# 添加engine和utils路径
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

# 导入推理引擎TecoInferEngine、优化文件路径PASS_PATH
from engine.tecoinfer_pytorch import TecoInferEngine
from engine.base import PASS_PATH

# 导入数据加载器load_data、预处理preprocess和后处理postprocess模块
from utils.datasets.image_classification_dataset import load_data
from utils.preprocess.pytorch.classification import preprocess
from utils.postprocess.pytorch.classification import postprocess


if __name__ == "__main__":
    # 动态shape的onnx文件需要指定运行时模型的输入shape, 按照模型输入设置, 注意batch=1
    input_size = [[1, 3, shape, shape]]
    
    # 初始化模型，支持onnx/tensorrt/tvm
    pipeline = TecoInferEngine(ckpt=ckpt,				# 模型的onnx文件路径
                               input_name=input_name,	# 导出模型onnx时的input_name
                               target=target,			# 可选:'onnx'进行onnxruntime-cpu推理、'sdaa'进行TecoInferenceEngine推理、'cuda'(paddle模型为'gpu')进行tensorRT推理
                               batch_size=batch_size,	# 数据集推理的batch_size
                               input_size=input_size,	# 指定初始化时的输入shape
                               dtype="float16", 		# 可选"float16"和"float32"，推荐"float16"
                               pass_path=pass_path,		# 推理框架优化文件，新适配模型设置为：PASS_PATH / "default_pass.py" 即可
                              )
    # 加载单个样本数据，并做预处理
    # 需要将输入数据处理为np.ndarray或按照输入顺序处理为[np.ndarray, np.ndarray, ...]格式
    input_data = load_data(demo_path, demo_infer=True)

    # 进行推理，输出为numpy格式数据
    prec = pipeline(images)

    # 后处理, 需要处理为可读的输出形式，例如目标检测算法打印坐标位置、种类和置信度, 分类模型打印出类别和score
    result = postprocess(prec)

    # 打印输出
    print(f"{demo_path}: {result}")
```

`example_multi_batch.py`：文件夹推理(batch=1)，在单个样本推理的基础上添加文件遍历即可，可以参考[resnet](../example/classification/resnet/example_multi_batch.py)补充。。

```
......
if __name__ == "__main__":
    
    ......

    for file_name in os.listdir(opt.data_path):
        file_path = os.path.join(opt.data_path, file_name)

        input_data = load_data(file_path, demo_infer=True)

        ......
```     

### 3.4 README文档添加
文档格式可参考模板[resnet](../example/classification/resnet/README.md)，各章节需要严格对齐，必须包含以下内容：

```
# 算法名称
## 1. 模型概述
    对模型进行简介
## 2. 快速开始
    使用当前模型推理的主要流程，可直接复制模板。
### 2.1 基础环境安装
    使用当前模型推理的基础环境说明，可直接复制模板。
### 2.2 安装第三方依赖
    介绍第三方依赖安装，可直接复制模板, 注意修改模型路径。
### 2.3 获取ONNX文件
    提供导出onnx文件的方法，包括：权重下载、导出代码、导出命令相关参数说明
    注：如果需要, PyTorch或PaddlePaddle模型源码, 放在 export_onnx.py 或同级目录下。
### 2.4 获取数据集
    提供所用数据集下载链接和处理代码，确保用户可根据此处说明获取可用数据集。
### 2.5 启动推理
    提供单个样本和文件夹推理命令行，以及推理结果和推理参数说明（参考resnet/README.md）
### 2.6 精度验证
    提供数据集推理命令行、推理结果和推理结果说明（参考resnet/README.md）
```

### 3.5 极限性能测试
功能说明：随机初始化构造输入，获取模型特定batch_size+shape下的极限性能。
适配内容：参考[模板](../tecoexec/testcase_configs/tecoexec_config.yaml) 添加配置文件至`TecoInference/contrib/tecoexec/tecoexec_config.yaml`，并参考[文档](../tecoexec/README.md)完成功能测试即可。

## 4. 完成测试并提交代码
适配完成后需要完成精度pipeline和极限性能的所有case测试，通过标准：

- 所有case：常用或源码默认shape + batch_size=1~max_batchsize(如此扩大shape即可1、4、8、16...，三SPA：1、3、6、12...)。

- 精度pipeline：TecoInferenceEngine的metric结果和onnxruntime-cpu的metric结果相对误差不超过0.1%。

- 极限性能：完成功能测试。

完成所有测试并通过后，参考[PR提交规范](./PullRequests.md)提交代码。