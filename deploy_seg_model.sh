python3 ./tools/deploy.py \
    ./configs/mmseg/segmentation_tensorrt-fp16_static-400x640.py \
    $MMSEG_DIR/configs/bisenetv2/bisenetv2_fcn_160k_adam-400x640.py \
    $MMSEG_DIR/work_dirs/bisenetv2_fcn_4xb4-amp-160k_adam-400x640/best_mIoU_iter_20000.pth \
    $MMSEG_DIR/data/adam/valid/023_jpg.rf.c6cb88f38eba73fdfd69d4d4cbf07298_mask.png \
    --work-dir ./work_dir/mmseg/bisenetv2/bisenetv2_fcn_400x640_adam \
    --show \
    --device cuda:0 \
    --dump-info
