export TORCH_SDAA_AUTOLOAD=cuda_migrate
# TORCH_SDAA_LOG_LEVEL=debug python main_resnet50_scratch.py --batch 64 --num-tasks 1 --learning-rate 2e-2 2>&1 | tee sdaa.log
python main_resnet50_scratch.py --batch 64 --num-tasks 1 --learning-rate 2e-2 2>&1 | tee sdaa.log