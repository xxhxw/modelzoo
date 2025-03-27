import onnx
import onnxsim
from onnxconverter_common import float16

import torch
import torch.nn as nn
import torchvision

import sys
# sys.path.append('/mnt/nvme/common/user_data/yqw/yolov9m')

from models.common import DetectMultiBackend

COCO_PATH = '/mnt/nvme/common/user_data/yqw/yolov9m/coco.yaml'
PT_PATH = '/mnt/nvme/common/user_data/yqw/yolov9m/yolov9-m-converted.pt'

# 初始化修改后的模型
model = DetectMultiBackend(PT_PATH, device='cpu', dnn=False, data=COCO_PATH, fp16=False)
model.eval()

# 创建虚拟输入
dummy_input = torch.randn(1, 3, 640, 640)

# 动态shape导出
dynamic_dims = {'input': {0: 'batch', 2: 'height', 3: 'width'},
                'output': {0: 'batch'}}

# 导出为ONNX格式
torch.onnx.export(model,
                  dummy_input,
                  "yolov9m_dyn.onnx",
                  opset_version=12,         	# ONNX opset版本
                  input_names=['input'],    	# 输入名称
                  output_names=['output'],  	# 输出名称
                  do_constant_folding=True, 	# 是否执行常量折叠优化
                  dynamic_axes=dynamic_dims)	# 动态shape

# 检查导出的ONNX模型
model_onnx = onnx.load("yolov9m_dyn.onnx")  # 加载ONNX模型
onnx.checker.check_model(model_onnx)  # 检查ONNX模型

# 简化ONNX模型
model_onnx, check = onnxsim.simplify(model_onnx)
assert check, '简化模型检查失败'

# 转换模型到float16
model_onnx = float16.convert_float_to_float16(model_onnx)
onnx.save(model_onnx, "yolov9m_dyn_float16.onnx")  # 保存转换后的模型

# 静态shape导出
torch.onnx.export(model,
                  dummy_input,
                  "yolov9m.onnx",
                  opset_version=12,         	# ONNX opset版本
                  input_names=['input'],    	# 输入名称
                  output_names=['output'],  	# 输出名称
                  do_constant_folding=True, 	# 是否执行常量折叠优化
                  dynamic_axes=None,	# 是否使用动态shape，不使用默认为None
                 )

