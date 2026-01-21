python3 ./tools/deploy.py \
    ./configs/mmseg/segmentation_tensorrt-fp16_static-400x640.py \
    $MMSEG_DIR/configs/pspnet/pspnet_r50-d8_4xb2-20e_adam-400x640.py \
    $MMSEG_DIR/work_dirs/pspnet_r50-d8_4xb2-20e_adam-512x1024/iter_2000.pth \
    $MMSEG_DIR/data/adam/leftImg8bit/test/dataset\ 2026-01-20\ 17-32-30/273_leftImg8bit.png \
    --work-dir ./work_dir/mmseg/pspnet/pspnet_r50_d8_400x640_adam \
    --show \
    --device cuda:0 \
    --dump-info
