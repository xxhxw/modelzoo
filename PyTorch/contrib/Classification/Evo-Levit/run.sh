export TORCH_SDAA_AUTOLOAD=cuda_migrate

torchrun --nproc_per_node=1  main_levit.py     --model EvoLeViT_256_384     --input-size 384     --batch-size 64     --data-path /data/dataset/imagenet     --output_dir /data/application/zhaohr/shipei/Evo-ViT/output_dir 2>&1 | tee sdaa.log