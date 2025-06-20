export TORCH_SDAA_AUTOLOAD=cuda_migrate
export SDAA_VISIBLE_DEVICES=8,9,10,11
rm -rf ./work_space/* 
mkdir ./work_space/history && mkdir ./work_space/log && mkdir ./work_space/models && mkdir ./work_space/save

python train.py 2>&1 | tee sdaa.log