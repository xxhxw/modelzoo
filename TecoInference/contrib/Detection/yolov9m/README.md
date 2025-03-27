# yolov9m bug复现

## 1. 导出onnx

```
conda activate tvm-build
cd <modelzoo_dir>/TecoInference/contrib/Detection/yolov9m
pip install -r requirements.txt
python export_onnx.py
```

## 2. 推理测试
为加快速度，只测试100个batch

### 1. 推理，cpu，fp32，推理通，指标正常
```
python example_valid.py --batch_size 4 --target onnx --dtype float32 --half False --ckpt yolov9m_dyn.onnx
```

### 2. 推理，cpu，fp16，推理通，指标正常
```
python example_valid.py --batch_size 4 --target onnx --dtype float16 --half True --ckpt yolov9m_dyn_float16.onnx
```

### 3. 推理，sdaa，fp32，推理通，指标异常
```
python example_valid.py --batch_size 4 --target sdaa --dtype float32 --half False --ckpt yolov9m_dyn.onnx
```

### 4. 推理，sdaa，fp16，推理报错，无指标
```
python example_valid.py --batch_size 4 --target sdaa --dtype float16 --half True --ckpt yolov9m_dyn_float16.onnx
```

