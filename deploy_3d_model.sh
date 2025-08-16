python3 tools/deploy.py ./configs/mmdet3d/monocular-detection/fcos3d_adam_tensorrt_static-416x640.py \
 $MMDET3D_DIR/work_dirs/fcos3d_regnet_adam/fcos3d_regnet_fpn_head-gn_3x_adam-mono3d.py \
 $MMDET3D_DIR/work_dirs/fcos3d_regnet_adam/epoch_36.pth \
 $HOME/Downloads/guilder_dataset/images/frame_000190.jpg \
 --work-dir ./work_dir/mmdet3d/fcos3d \
 --device cuda:0 \
 --show

