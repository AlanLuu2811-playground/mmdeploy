python3 ./tools/deploy.py \
    ./configs/mmdet/instance-seg/instance-seg_rtmdet-ins_tensorrt_static-640x640.py \
    /home/alan_khang/dev/YOLOF-MaskV2-mmcv/configs/unipercepnet_s_regnetx_4gf_se_sam_3x_coco.py \
    /home/alan_khang/dev/YOLOF-MaskV2-mmcv/work_dirs/unipercepnet_s_regnetx_4gf_se_sam_3x_coco/best_coco_segm_mAP_epoch_18.pth \
    /home/alan_khang/dev/mmdetection/demo/demo.jpg \
    --work-dir ./work_dir/mmdet/unipercepnet_s_regnetx_4gf_se_sam_coco \
    --show \
    --device cuda:0 \
    --dump-info
