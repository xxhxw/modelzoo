# YOLOV9-M

## 1. 模型概述

YOLOv9-M是YOLO系列模型中的一种中型版本，结合了高效性和准确性的特点。与先前版本相比，YOLOv9-M在网络结构上进行了进一步的优化，引入了改进的特征金字塔网络（FPN）和跨阶段局部网络（CSPNet），提升了特征提取和多尺度检测的能力。此外，YOLOv9-M采用了更高效的注意力机制，增强了对小目标和复杂场景的检测性能，同时保持了较低的计算成本。该模型在速度与精度之间取得了较好的平衡，适用于对实时性和检测准确性均有要求的应用场景。

## 2. 快速开始

使用本模型执行模型推理的主要流程如下：
1. 基础环境安装：介绍推理前需要完成的基础环境检查和安装。
2. 安装第三方依赖：介绍如何安装模型推理所需的第三方依赖。
3. 获取ONNX文件：介绍如何获取推理所需的ONNX模型文件。
4. 获取数据集：介绍如何获取推理所需的数据集。
5. 启动推理：介绍如何运行推理。
6. 精度验证：介绍如何验证推理精度。

### 2.1 基础环境安装

请参考推理首页的[基础环境安装](../../../README.md)章节，完成推理前的基础环境检查和安装。

### 2.2 安装第三方依赖

1. 执行以下命令，进入conda环境。
   ```
   conda activate tvm-build
   ```

2. 执行以下命令，进入第三方依赖安装脚本所在目录。

   ```
   cd <modelzoo_dir>/TecoInference/contrib/dedtection/yolov9m
   ```

3. 执行以下命令，安装第三方依赖。

   ```
   pip install -r requirements.txt
   ```

   **注意**：若速度过慢，可加上`-i`参数指定源。


### 2.3 获取ONNX文件
1. 下载[权重文件](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2FWongKinYiu%2Fyolov9%2Freleases%2Fdownload%2Fv0.1%2Fyolov9-m-converted.pt)
，存放到权重目录中(例如/mnt/nvme/common/user_data/yqw/yolov9-m-converted.pt)

2. 执行以下命令，导出模型的ONNX文件。

   
   ```
   python export_onnx.py --ckpt /mnt/nvme/common/user_data/yqw/yolov9-m-converted.pt
   ```
   
   此命令运行结束后会生成三个onnx文件，分别是yolov9m.onnx，yolov9m_dyn.onnx，yolov9m_dyn_float16.onnx.onnx; 分别代表静态shape、动态shape、float16精度的动态shape格式的onnx。在使用中请使用 yolov9m_dyn_float16.onnx 进行推理。

### 2.4 获取数据集

您可以通过以下方式获取推理所需的数据集：
- 推理可使用内置的demo数据样本。Demo数据样本位于仓库的`./images`目录。
- 下载 COCO2017数据集 ([train](http://images.cocodataset.org/zips/train2017.zip), [val](http://images.cocodataset.org/zips/val2017.zip), [test](http://images.cocodataset.org/zips/test2017.zip)) and [labels](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-segments.zip). 

### 2.5 启动推理

1. 在Docker环境中，进入推理脚本所在目录。

   ```
   cd <modelzoo_dir>/TecoInference/example/detection/yolov9m
   ```

2. 运行推理。

   - 单张图像推理：使用单张图像作为输入，进行推理。
   
     ```
     python example_single_batch.py  --ckpt yolov9m_dyn.onnx --batch_size 1 --input_name input --target onnx --data_config coco.yaml  --data_path ./images/room.jpg --dtype float32
     ```

      推理结果：

      ```
      目标 1: - 类别: vase - 坐标: (242.0, 191.0, 254.0, 213.0) - 置信度: 0.52
      目标 2: - 类别: person - 坐标: (384.0, 152.0, 401.0, 205.0) - 置信度: 0.48
      目标 3: - 类别: potted plant - 坐标: (231.0, 151.0, 267.0, 214.0) - 置信度: 0.47
      目标 4: - 类别: vase - 坐标: (361.0, 215.0, 374.0, 241.0) - 置信度: 0.42
      目标 5: - 类别: chair - 坐标: (364.0, 223.0, 415.0, 361.0) - 置信度: 0.32
      目标 6: - 类别: tv - 坐标: (474.0, 98.0, 526.0, 157.0) - 置信度: 0.29
      目标 7: - 类别: clock - 坐标: (5.0, 0.0, 174.0, 87.0) - 置信度: 0.28
      目标 8: - 类别: chair - 坐标: (415.0, 224.0, 438.0, 291.0) - 置信度: 0.26
      目标 9: - 类别: wine glass - 坐标: (314.0, 182.0, 324.0, 215.0) - 置信度: 0.25
      目标 10: - 类别: chair - 坐标: (216.0, 243.0, 296.0, 349.0) - 置信度: 0.20
      ```

    - 文件夹推理：使用文件夹，对文件中的所有图像进行推理。

      ```
      python example_multi_batch.py  --ckpt yolov9m_dyn.onnx --batch_size 1 --input_name input --target onnx --data_config coco.yaml  --data_path ./images --dtype float32
      ```

      推理结果：
  
      ```
      第1张图片：room.jpg
      目标 1: - 类别: vase - 坐标: (242.0, 191.0, 254.0, 213.0) - 置信度: 0.52
      目标 2: - 类别: person - 坐标: (384.0, 152.0, 401.0, 205.0) - 置信度: 0.48
      目标 3: - 类别: potted plant - 坐标: (231.0, 151.0, 267.0, 214.0) - 置信度: 0.47
      目标 4: - 类别: vase - 坐标: (361.0, 215.0, 374.0, 241.0) - 置信度: 0.42
      目标 5: - 类别: chair - 坐标: (364.0, 223.0, 415.0, 361.0) - 置信度: 0.32
      目标 6: - 类别: tv - 坐标: (474.0, 98.0, 526.0, 157.0) - 置信度: 0.29
      目标 7: - 类别: clock - 坐标: (5.0, 0.0, 174.0, 87.0) - 置信度: 0.28
      目标 8: - 类别: chair - 坐标: (415.0, 224.0, 438.0, 291.0) - 置信度: 0.26
      目标 9: - 类别: wine glass - 坐标: (314.0, 182.0, 324.0, 215.0) - 置信度: 0.25
      目标 10: - 类别: chair - 坐标: (216.0, 243.0, 296.0, 349.0) - 置信度: 0.20

      第2张图片：bear.jpg
      目标 1: - 类别: teddy bear - 坐标: (0.0, 67.0, 586.0, 639.0) - 置信度: 0.53
      ```

 模型推理参数说明：

| 参数 | 说明 | 默认值 |
| ------------- | ------------- | ------------- |
| data_path  |数据路径 |./images/room.jpg|
| ckpt       | 模型onnx路径  | ./yolov9m_dyn.onnx |
| batch_size | 推理的batch_size  | 1 |
| input_size | 推理输入尺寸  | 640 |
| target     | 推理的设备 | `onnx` |
| input_name | onnx输入的名称 | input |
| dtype      | 模型推理的数据类型 | float32 |
| conf_thres | NMS的置信度阈值 | 0.2 |
| data_config| 数据集配置文件  | ./coco.yaml |


### 2.6 精度验证

请提前准备好COCO2017数据集，执行以下命令，获得推理精度数据。

```
python example_valid.py --ckpt yolov9m_dyn.onnx --batch_size 4 --input_name input --target onnx --data_config coco.yaml  --data_path /mnt/nvme/common/inference_dataset/coco/val2017.txt --dtype float32
```

精度结果如下：

```
mAP@0.5~0.95: 0.502858
summary: avg_sps: 11.075851719506225, e2e_time: 452.1122546195984, avg_inference_time: 0.320773968639144, avg_preprocess_time: 0.007225121879577637, avg_postprocess: 0.03352416534423828
```

 结果说明：

| 参数 | 说明 |
| ------------- | ------------- |
| avg_sps | 吞吐量(images/s) |
| e2e_time | 端到端总耗时(s)  |
| avg_inference_time | 平均推理计算时间(s)  |
| avg_preprocess_time     | 平均预处理时间(s)  |
| avg_postprocess |  平均后处理时间(s) |
| mAP50-95      | 数据集验证精度  |