# ResNet50



## 模型概述
ResNet-50是一种深度卷积神经网络模型，采用了残差网络（ResNet）的结构，通过引入“残差块”（Residual Block）来解决了深度神经网络训练中的梯度消失和表示瓶颈问题。ResNet-50模型在各种计算机视觉任务中表现优异，如图像分类、目标检测和语义分割等。由于其良好的性能和广泛的应用，ResNet-50已成为深度学习和计算机视觉领域的重要基础模型之一。


## Quick Start Guide

### 1、环境准备

#### 1.1.拉取代码仓

```
git clone http://10.10.30.109/tecoap/modelzoo.git
```

#### 1.2.Docker 环境准备 
##### 1.2.1创建Docker环境
- 进入Dockerfile所在目录，运行以下命令
```
cd <modelzoo-dir>/PaddlePaddle/Classification/ResNet
DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0 docker build . -t paddle_r50
```
镜像内部软件栈版本信息如下:
[SDAA软件栈版本信息](../../../.dependencies.json)
|软件名|paddle|sdaadriver|sdaart|tccl|dnn|blas|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|版本信息|1.0.0|1.0.0|1.0.0|1.14.0|1.15.0|1.15.0|

- 创建ResNet50 PaddlePaddle sdaa docker容器

```
docker run  -itd --name r50_pd -v <dataset_path>:/imagenet  --net=host --ipc=host --device /dev/tcaicard0 --device /dev/tcaicard1 --device /dev/tcaicard2 --device /dev/tcaicard3 --shm-size=128g paddle_r50 /bin/bash
```

- 参数介绍详见[Docker configuration](./docs/Docker_configuration.md)

- 进入Docker 容器
```
docker exec -it r50_pd /bin/bash
```
##### 1.2.2 创建Teco虚拟环境
```
cd /workspace/Classification/ResNet/run_scripts
conda activate paddle_env
```

#### 1.3. Host 环境准备
```
  cd /PaddlePaddle/Classification/ResNet
  pip install -r requirements.txt
```

### 2、数据集准备
#### 2.1. 获取数据集

ResNet50运行在ImageNet 1k上，这是一个来自ILSVRC挑战赛的广受欢迎的图像分类数据集。要使用混合精度或FP32精度训练您的模型，请根据以下步骤获取并处理数据集：

2.1.1. 从公开网站中获取数据集下载
https://image-net.org/download-images

2.1.2. 从共享存储中获取
从应用平台部的共享存储上获取已经下载, 解压并处理好的ImageNet数据集
```
cp -r /mnt/dataset/imagenet ~
```
#### 2.2. 解压数据集

    - 解压训练数据集
    ```
    mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
    tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
    find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    cd ..
    ```
    - 解压测试数据集并将图像移动到子文件夹中
    ```
    mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
    wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
    ```
#### 2.3. 在本文档中，包含`train/`和`val/`目录被称为`path to imagenet`，数据集目录结构参考如下所示:
```
   ├── ImageNet2012
         ├──train
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...
              ├──类别2
                    │──图片1
                    │──图片2
                    │   ...
              ├──...
         ├──val
              ├──类别1
                    │──图片1
                    │──图片2
                    │   ...
              ├──类别2
                 │──图片1
                    │──图片2
                    │   ...
   ```

   > **说明：**
   > 该数据集的训练过程脚本只作为一种参考示例。



### 3、 启动训练
```
# Docker环境
cd /workspace/Classification/ResNet/run_scripts

# Host环境
cd /PaddlePaddle/Classification/ResNet/run_scripts

```
#### 3.1.该模型支持单机单卡、单机四卡、单机八卡、两机八卡

- Demo测试正确性

    下面给出了一个在单卡单核上20steps的训练脚本的例子，用于测试模型代码及数据集正确性，更多的训练参数可以参考[run_scripts/README.md](./run_scripts/README.md)
    ```
    python run_paddle_resnet.py --model_name resnet50 --nproc_per_node 1 --bs 64 --lr 0.064 --device sdaa --step 20 --epoch 1 --dataset_path /imagenet  --grad_scale True --autocast True
    ```
    - 出现`experiment ended` 则表示程序正常运行完毕

- 单机单卡（4核）组训练
    ```
    python run_paddle_resnet.py --model_name resnet50 --nproc_per_node 4 --bs 64 --lr 0.256 --device sdaa  --epoch 50 --dataset_path /imagenet  --grad_scale True --autocast True
    ```
    
- 单机四卡（4*4核）组训练
    ```
    python run_paddle_resnet.py --model_name resnet50  --nproc_per_node 16 --bs 64 --lr 1.024 --device sdaa  --epoch 50 --dataset_path /imagenet  --grad_scale True --autocast True

    ```

- 单机八卡（8*4核）组训练
    ```
    python run_paddle_resnet.py --model_name resnet50  --nproc_per_node 32 --bs 64 --lr 2.048 --device sdaa  --epoch 90 --dataset_path /imagenet  --grad_scale True --autocast True

    ```

- 两机八卡（8*4核）组训练
    ```
    # node0
    python run_paddle_resnet.py --model_name resnet50 --master_addr ip_address1,ip_address2 --master_port 29500 --nproc_per_node 16 --bs 64 --lr 2.048 --device sdaa  --epoch 50 --dataset_path /imagenet  --grad_scale True --autocast True --nnode 2

    # node1
    python run_paddle_resnet.py --model_name resnet50  --master_addr ip_address1,ip_address2 --master_port 29500 --nproc_per_node 16 --bs 64 --lr 2.048 --device sdaa  --epoch 50 --dataset_path /imagenet  --grad_scale True --autocast True --nnode 2

    #请将上述命令中的ip_address1, ip_address2 修改为所使用节点的ip地址，其中ip_address1为主节点
    ```

#### 3.2.模型训练脚本参数说明如下：

参数名 | 解释 | 样例
-----------------|-----------------|-----------------
model_name |模型名称 | --model_name resnet50
epoch| 训练轮次，和训练步数冲突 | --epoch 50
step | 训练步数，和训练轮数冲突 | --step 1
batch_size/bs | 每个rank的batch_size | --batch_size 64 / --bs 64
dataset_path | 数据集路径 | --dataset_path /imagenet
nproc_per_node | DDP时，每个node上的rank数量。不输入时，默认为1，跑单核 | --nproc_per_node 4
nnode | DDP时，node数量。不输入时，默认为1，跑单卡。| --nnode 2
node_rank|多机时，node的序号|--node_rank 0
master_addr|多机时，主节点的addr|--master_addr 127.0.0.1
master_port|多机时，主节点的端口号|--matser_port 13400
lr|学习率|--lr 0.064
num_workers|dataloader读取数据进程数|--num_workers 2
device|设备类型|--device cuda --device sdaa
autocast|开启autocast|--autocast True
grad_scaler| 使用grad_scale | --grad_scale True


* 注：当前已支持模型包括：resnet50、resnet18、resnet101，请在使用时将 `--model_name` 替换成对应模型

### 4、训练结果

| 芯片 |卡 |频率 | 模型 |  混合精度 |Batch size|Shape| 吞吐量| AccTop1|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|SDAA|1|2.0G| ResNet18 |是|256|224*224| 2020 img/s| 70.36% |
|SDAA|1|2.0G| ResNet50 |是|256|224*224| 888 img/s| 76.09% |
|SDAA|1|2.0G| ResNet101 |是|256|224*224| 536 img/s| 78.14% |


### 5、基于PaddleInference推理
#### 5.1.基于训练的模型导出推理的模型权重
(1) 依赖安装
```
cd modelzoo/PaddlePaddle/Classification/ResNet
pip install -r requirements.txt
pip install .
```
推理模型导出所需要的训练文件如下：
```
$ tree output/ResNet50/
ResNet50/
├── best_model.pdopt
├── best_model.pdparams
├── best_model.pdstates
```
(2) 从训练模型导出推理模型，运行如下命令：
参数名 | 解释
-----------------|-----------------|
-c|  指定训练时的配置文件 |
-o |   Global.pretrained_model 指定需要转换的训练模型
-o |     Global.save_inference_dir指定导出推理模型的保存路径
```
python tools/export_model.py \
    -c tools/ResNet50_amp_O1.yaml \
    -o Global.pretrained_model=output/ResNet50/best_model \
    -o Global.save_inference_dir=tools/infer_save
```
(3) 执行结果如下：
```
ppcls INFO: Export succeeded! The inference model exported has been saved in "tools/infer_save".
```
#### 5.2.基于导出的模型执行推理脚本
```
cd deploy
python3 python/predict_cls.py -c configs/inference_cls_sdaa.yaml
```
#### 5.3.推理结果
推理图片为‘黑琴鸡’，推理结果表示为‘黑琴鸡’的概率为0.74，符合预期
```
ILSVRC2012_val_00030010.jpeg:   class id(s): [80, 23, 93, 136, 100], score(s): [0.74, 0.09, 0.02, 0.02, 0.01], label_name(s): ['black grouse', 'vulture', 'hornbill', 'European gallinule, Porphyrio porphyrio', 'black swan, Cygnus atratus']
```