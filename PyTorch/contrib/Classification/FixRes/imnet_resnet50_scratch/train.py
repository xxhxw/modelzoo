# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted to tecorigin hardware
import os
import os.path as osp
from typing import Optional
import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
import attr
from torchvision import datasets
import torchvision.models as models
import numpy as np
from .config import TrainerConfig, ClusterConfig
from .transforms import get_transforms
from .samplers import RASampler

@attr.s(auto_attribs=True)
class TrainerState:
    """
    Contains the state of the Trainer.
    It can be saved to checkpoint the training and loaded to resume it.
    """

    epoch: int
    accuracy:float
    model: nn.Module
    optimizer: optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    scaler: torch.cuda.amp.GradScaler  # 新增GradScaler状态

    def save(self, filename: str) -> None:
        data = attr.asdict(self)
        # store only the state dict
        data["model"] = self.model.state_dict()
        data["optimizer"] = self.optimizer.state_dict()
        data["lr_scheduler"] = self.lr_scheduler.state_dict()
        data["scaler"] = self.scaler.state_dict()  # 保存scaler状态
        data["accuracy"] = self.accuracy
        torch.save(data, filename)

    @classmethod
    def load(cls, filename: str, default: "TrainerState") -> "TrainerState":
        data = torch.load(filename)
        # We need this default to load the state dict
        model = default.model
        model.load_state_dict(data["model"])
        data["model"] = model

        optimizer = default.optimizer
        optimizer.load_state_dict(data["optimizer"])
        data["optimizer"] = optimizer

        lr_scheduler = default.lr_scheduler
        lr_scheduler.load_state_dict(data["lr_scheduler"])
        data["lr_scheduler"] = lr_scheduler
        
        # 加载scaler状态
        scaler = default.scaler
        if "scaler" in data:
            scaler.load_state_dict(data["scaler"])
        data["scaler"] = scaler
        
        return cls(**data)


class Trainer:
    def __init__(self, train_cfg: TrainerConfig, cluster_cfg: ClusterConfig) -> None:
        self._train_cfg = train_cfg
        self._cluster_cfg = cluster_cfg

    def __call__(self) -> Optional[float]:
        """
        Called for each task.

        :return: The master task return the final accuracy of the model.
        """
        self._setup_process_group()
        self._init_state()
        final_acc = self._train()
        return final_acc

    def checkpoint(self, rm_init=True):
        save_dir = osp.join(self._train_cfg.save_folder, str(self._train_cfg.job_id))
        os.makedirs(save_dir, exist_ok=True)
        self._state.save(osp.join(save_dir, "checkpoint.pth"))
        self._state.save(osp.join(save_dir, "checkpoint_"+str(self._state.epoch)+".pth"))
        if rm_init:
            os.remove(self._cluster_cfg.dist_url[7:])  
        empty_trainer = Trainer(self._train_cfg, self._cluster_cfg)
        return empty_trainer

    def _setup_process_group(self) -> None:
        torch.cuda.set_device(self._train_cfg.local_rank)
        torch.distributed.init_process_group(
            backend=self._cluster_cfg.dist_backend,
            init_method=self._cluster_cfg.dist_url,
            world_size=self._train_cfg.num_tasks,
            rank=self._train_cfg.global_rank,
        )
        print(f"Process group: {self._train_cfg.num_tasks} tasks, rank: {self._train_cfg.global_rank}")

    def _init_state(self) -> None:
        """
        Initialize the state and load it from an existing checkpoint if any
        """
        torch.manual_seed(0)
        np.random.seed(0)
        print("Create data loaders", flush=True)
        
        Input_size_Image=self._train_cfg.input_size
        
        Test_size=Input_size_Image
        print("Input size : "+str(Input_size_Image))
        print("Test size : "+str(Input_size_Image))
        print("Initial LR :"+str(self._train_cfg.lr))
        
        transf=get_transforms(input_size=Input_size_Image,test_size=Test_size, kind='full', crop=True, need=('train', 'val'), backbone=None)
        transform_train = transf['train']
        transform_test = transf['val']
        
        train_set = datasets.ImageFolder(self._train_cfg.imnet_path + '/train',transform=transform_train)
        train_sampler = RASampler(
            train_set,self._train_cfg.num_tasks,self._train_cfg.global_rank,len(train_set),self._train_cfg.batch_per_gpu,repetitions=3,len_factor=2.0,shuffle=True, drop_last=False
        )
        self._train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self._train_cfg.batch_per_gpu,
            num_workers=(self._train_cfg.workers-1),
            sampler=train_sampler,
        )
        test_set = datasets.ImageFolder(self._train_cfg.imnet_path  + '/val',transform=transform_test)

        self._test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=self._train_cfg.batch_per_gpu, shuffle=False, num_workers=(self._train_cfg.workers-1),
        )

        print(f"Total batch_size: {self._train_cfg.batch_per_gpu * self._train_cfg.num_tasks}", flush=True)

        print("Create distributed model", flush=True)
        model = models.resnet50(pretrained=False)
        
        model.cuda(self._train_cfg.local_rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self._train_cfg.local_rank], output_device=self._train_cfg.local_rank
        )
        linear_scaled_lr = 8.0 * self._train_cfg.lr * self._train_cfg.batch_per_gpu * self._train_cfg.num_tasks /512.0
        optimizer = optim.SGD(model.parameters(), lr=linear_scaled_lr, momentum=0.9,weight_decay=1e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30)
        
        # 初始化GradScaler
        scaler = torch.cuda.amp.GradScaler()
        
        self._state = TrainerState(
            epoch=0,accuracy=0.0, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, scaler=scaler
        )
        checkpoint_fn = osp.join(self._train_cfg.save_folder, str(self._train_cfg.job_id), "checkpoint.pth")
        if os.path.isfile(checkpoint_fn):
            print(f"Load existing checkpoint from {checkpoint_fn}", flush=True)
            self._state = TrainerState.load(checkpoint_fn, default=self._state)

    def _train(self) -> Optional[float]:
        criterion = nn.CrossEntropyLoss()
        print_freq = 10
        acc = None
        max_accuracy=0.0
        # Start from the loaded epoch
        start_epoch = self._state.epoch
        for epoch in range(start_epoch, self._train_cfg.epochs):
            print(f"Start epoch {epoch}", flush=True)
            self._state.model.train()
            self._state.lr_scheduler.step(epoch)
            self._state.epoch = epoch
            running_loss = 0.0
            count=0
            for i, data in enumerate(self._train_loader):
                inputs, labels = data
                inputs = inputs.cuda(self._train_cfg.local_rank, non_blocking=True)
                labels = labels.cuda(self._train_cfg.local_rank, non_blocking=True)

                # 使用AMP进行前向传播
                with torch.cuda.amp.autocast():
                    outputs = self._state.model(inputs)
                    loss = criterion(outputs, labels)

                # 使用scaler进行反向传播
                self._state.optimizer.zero_grad()
                self._state.scaler.scale(loss).backward()
                self._state.scaler.step(self._state.optimizer)
                self._state.scaler.update()

                running_loss += loss.item()
                count=count+1
                if i % print_freq == print_freq - 1:
                    print(f"[{epoch:02d}, {i:05d}] loss: {running_loss/print_freq:.3f}", flush=True)
                    running_loss = 0.0
                if count>=5005 * 512 /(self._train_cfg.batch_per_gpu * self._train_cfg.num_tasks):
                    break
                
            if epoch==self._train_cfg.epochs-1:
                print("Start evaluation of the model", flush=True)
                
                correct = 0
                total = 0
                count=0.0
                running_val_loss = 0.0
                self._state.model.eval()
                with torch.no_grad():
                    for data in self._test_loader:
                        images, labels = data
                        images = images.cuda(self._train_cfg.local_rank, non_blocking=True)
                        labels = labels.cuda(self._train_cfg.local_rank, non_blocking=True)
                        
                        # 在验证时也使用AMP
                        with torch.cuda.amp.autocast():
                            outputs = self._state.model(images)
                            loss_val = criterion(outputs, labels)
                        
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        running_val_loss += loss_val.item()
                        count=count+1.0

                acc = correct / total
                ls_nm=running_val_loss/count
                print(f"Accuracy of the network on the 50000 test images: {acc:.1%}", flush=True)
                print(f"Loss of the network on the 50000 test images: {ls_nm:.3f}", flush=True)
                self._state.accuracy = acc
                if self._train_cfg.global_rank == 0:
                    self.checkpoint(rm_init=False)
                print("accuracy val epoch "+str(epoch)+" acc= "+str(acc))
                max_accuracy=np.max((max_accuracy,acc))
                if epoch==self._train_cfg.epochs-1:
                    return acc