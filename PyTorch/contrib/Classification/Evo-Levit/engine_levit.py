# modified from https://github.com/facebookresearch/LeViT
"""
Train and eval functions used in main.py
"""
# Adapted to tecorigin hardware
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from levit.losses_levit import DistillationLoss
import utils
from torch.sdaa import amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# from levit.evo_levit import set_learn_tradeoff

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, 
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    scaler=None):  # 新增scaler参数
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    mode1 = False
    mode2 = False
    mode4 = False
    CTPEP = False

    # 初始化默认scaler（保持向后兼容）
    if scaler is None:
        scaler = torch.sdaa.amp.GradScaler()

    # 模型模式设置
    if epoch < 200:
        # 获取原始模型（兼容DDP和单卡模式）
        # print("Available methods:", dir(model.module if hasattr(model, 'module') else model))
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model.stage_wise_prune = False
        actual_model.set_learn_tradeoff(False)
    else:
        actual_model = model.module if hasattr(model, 'module') else model
        actual_model.stage_wise_prune = True
        actual_model.set_learn_tradeoff(True)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Mixup增强
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        # 自动混合精度上下文
        with torch.sdaa.amp.autocast():
            try:
                # 确保输入张量连续
                samples = samples.contiguous()
                
                if mode1 or mode4:
                    outputs = model(samples, epoch)
                elif mode2 or CTPEP:
                    outputs = model(samples, epoch)
                else:
                    # print("Using default forward pass")
                    outputs = model(samples)
                
                loss = criterion(samples, outputs, targets)
            except RuntimeError as e:
                print(f"Forward pass failed: {str(e)}")
                print(f"Input shape: {samples.shape}")
                print(f"Current GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
                raise

        # 检查损失值有效性
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        # 梯度计算与更新
        optimizer.zero_grad()
        
        # 使用scaler进行梯度缩放
        scaler.scale(loss).backward()
        
        # 梯度裁剪（如果需要）
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, norm_type=2.0)
        
        # 参数更新
        scaler.step(optimizer)
        scaler.update()

        # 同步和EMA更新
        torch.sdaa.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # 记录指标
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # 手动释放内存
        del outputs, loss
        torch.sdaa.empty_cache()

    # 同步所有进程的指标
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    mode1 = False
    mode2 = False
    mutual = False  # True
    cls = True
    mode4 = False
    CTPEP = False  # False

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if mode1:
                output = model(images, epoch)
                acc1, acc5 = accuracy(output[1], target, topk=(1, 5))
                acc12, acc52 = accuracy(output[2], target, topk=(1, 5))
                acc13, acc53 = accuracy(output[3], target, topk=(1, 5))
                print("net 2 accuracy: {}, {}, net 3 accuracy: {}, {}, net 4 accuracy: {}, {}".format(acc1.item(),
                                                                                                      acc5.item(),
                                                                                                      acc12.item(),
                                                                                                      acc52.item(),
                                                                                                      acc13.item(),
                                                                                                      acc53.item()))
                output = output[0]
            elif mode2:
                output = model(images, epoch)
                acc1, acc5 = accuracy(output[1], target, topk=(1, 5))
                acc12, acc52 = accuracy(output[2], target, topk=(1, 5))
                acc13, acc53 = accuracy(output[3], target, topk=(1, 5))
                acc14, acc54 = accuracy(output[4], target, topk=(1, 5))
                print(
                    "net 2 accuracy: {}, {}, net 3 accuracy: {}, {}, net 4 accuracy: {}, {}, net merge accuracy: {}, {}".format(
                        acc1.item(), acc5.item(), acc12.item(), acc52.item(), acc13.item(), acc53.item(), acc14.item(),
                        acc54.item()))
                output = output[0]
            elif mode4:
                output = model(images, epoch)
                acc1, acc5 = accuracy(output[1], target, topk=(1, 5))
                acc12, acc52 = accuracy(output[2], target, topk=(1, 5))
                print("net 2 accuracy: {}, {}, net 3 accuracy: {}, {}".format(acc1.item(), acc5.item(), acc12.item(),
                                                                              acc52.item()))
                output = output[0]
            elif mutual:
                output = model(images)
                acc1, acc5 = accuracy(output[1], target, topk=(1, 5))
                print("net depth accuracy: {}, {}".format(acc1, acc5))
                output = output[0]
            elif cls:
                if CTPEP:
                    output = model(images, epoch)
                else:
                    output = model(images)

                # print("net cls accuracy: {}, {}".format(acc1, acc5))
                # output = output[0]
            else:
                output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print(output.mean().item(), output.std().item())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
