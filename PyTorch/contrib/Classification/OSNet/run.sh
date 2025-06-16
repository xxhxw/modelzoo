SDAA_VISIBLE_DEVEICES=12,13,14,15 python scripts/main.py \
--config-file configs/im_osnet_x1_0_softmax_256x128_amsgrad_cosine.yaml \
--transforms random_flip random_erase \
--root /data/application/zhaohr/shipei/deep-person-reid/ 2>&1 | tee sdaa.log