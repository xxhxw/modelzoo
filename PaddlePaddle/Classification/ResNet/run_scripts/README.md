## 运行样例

### 参数介绍
参数名 | 解释 | 样例 
-----------------|-----------------|----------------- 
model_name |模型名称 | --model_name resnet50
epoch| 训练轮次，和训练步数冲突 | --epoch 50
step | 训练步数，和训练轮数冲突 | --step 1
batch_size/bs | 每个rank的batch_size | --batch_size 32 / --bs 32
dataset_path | 数据集路径 | --dataset_path
nproc_per_node | DDP时，每个node上的rank数量。不输入时，默认为1，跑单核 | --nproc_per_node 4
nnode | DDP时，node数量。不输入时，默认为1，跑单卡。| --nnode 2
node_rank|多机时，node的序号|--node_rank 0
master_addr|多机时，主节点的addr|--master_addr 127.0.0.1
master_port|多机时，主节点的端口号|--matser_port 13400
precision_align|开启逐层精度对齐|--precision_align True
precision_align_cuda_path|指定逐层精度对齐cuda-golden基准数据。|--precision_align_cuda_path compare_results_all_r50_pd
precision_align_log_path|指定逐层精度对齐日志输出结果。|--precision_align_log_path precision_align_log/r50_pd
lr|学习率|--lr 1e-3
num_workers|dataloader读取数据进程数|--num_workers 2
device|设备类型|--device cuda --device sdaa
profiler|开启profile|--profile True
autocast|开启autocast|--autocast True
grad_scaler| 使用grad_scale | --grad_scale True


### Resnet50


#### 功能测试

1. 单核组
```
python run_paddle_resnet.py --model_name resnet50 --nproc_per_node 1 --bs 64 --lr 0.064 --device sdaa --step 20 --epoch 1 --dataset_path /imagenet/  --grad_scale True --autocast True
```

2. 单卡/多卡
```
单卡
python run_paddle_resnet.py --model_name resnet50 --nproc_per_node 4 --bs 64 --lr 0.256 --device sdaa --step 20 --epoch 1 --dataset_path /imagenet/  --grad_scale True --autocast True
四卡
python run_paddle_resnet.py --model_name resnet50  --nproc_per_node 16 --bs 64 --lr 1.024 --device sdaa --step 20 --epoch 1 --dataset_path /imagenet/  --grad_scale True --autocast True
```


3. 逐层精度
```
python run_paddle_resnet.py --model_name resnet50 --nproc_per_node 1 --bs 64 --lr 0.064 --device sdaa --step 2 --epoch 1 --dataset_path /imagenet/  --grad_scale True  --precision_align True  --precision_align_cuda_path ./log  --precision_align_log_path ./log
```


4. profiler
```
python run_paddle_resnet.py --model_name resnet50 --nproc_per_node 1 --bs 64 --lr 0.064 --device sdaa --step 20 --epoch 1 --dataset_path /imagenet/  --grad_scale True  --autocast True --profiler True  
```

5. 两机八卡
```
node0
python run_paddle_resnet.py --model_name resnet50  --master_addr 10.10.7.7,10.10.6.18 --master_port 29500 --nproc_per_node 16 --bs 64 --lr 2.048 --device sdaa --step 20 --epoch 1 --dataset_path /imagenet/  --grad_scale True --autocast True

node1
python run_paddle_resnet.py --model_name resnet50  --master_addr 10.10.7.7,10.10.6.18 --master_port 29500 --nproc_per_node 16 --bs 64 --lr 2.048 --device sdaa --step 20 --epoch 1 --dataset_path /imagenet/  --grad_scale True --autocast True
```

#### 长训
1. 单卡/多卡
```
单卡
python run_paddle_resnet.py --model_name resnet50 --nproc_per_node 4 --bs 64 --lr 0.256 --device sdaa  --epoch 50 --dataset_path /imagenet/  --grad_scale True --autocast True

单机四卡
python run_paddle_resnet.py --model_name resnet50  --nproc_per_node 16 --bs 64 --lr 1.024 --device sdaa  --epoch 50 --dataset_path /imagenet/  --grad_scale True --autocast True

单机八卡
python run_paddle_resnet.py --model_name resnet50  --nproc_per_node 32 --bs 64 --lr 2.048 --device sdaa  --epoch 90 --dataset_path /imagenet  --grad_scale True --autocast True
```
2. 两机八卡
```
node0
python run_paddle_resnet.py --model_name resnet50 --master_addr ip_address1,ip_address2 --master_port 29500 --nproc_per_node 16 --bs 64 --lr 2.048 --device sdaa  --epoch 50 --dataset_path /imagenet/  --grad_scale True --autocast True --nnode 2

node1
python run_paddle_resnet.py --model_name resnet50  --master_addr ip_address1,ip_address2 --master_port 29500 --nproc_per_node 16 --bs 64 --lr 2.048 --device sdaa  --epoch 50 --dataset_path /imagenet/  --grad_scale True --autocast True --nnode 2
#请将上述命令中的ip_address1, ip_address2 修改为所使用节点的ip地址，其中ip_address1为主节点
```
* 1、多机任务时，请确保master_addr中主节点在前