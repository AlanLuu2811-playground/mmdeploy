python3 tools/deploy.py ./configs/mmdet3d/monocular-detection/fcos3d_adam_tensorrt_static-416x640.py \
 /home/alan_khang/dev/mmdetection3d/work_dirs/fcos3d_regnet_adam/fcos3d_regnet_fpn_head-gn_3x_adam-mono3d.py \
 /home/alan_khang/dev/mmdetection3d/work_dirs/fcos3d_regnet_adam/epoch_36.pth \
 /home/alan_khang/Downloads/guilder_dataset/images/frame_000190.jpg \
 --work-dir ./work_dir/mmdet3d/fcos3d \
 --device cuda:0 \
 --show

