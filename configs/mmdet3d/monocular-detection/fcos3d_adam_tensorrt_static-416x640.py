_base_ = ['./monocular-detection_tensorrt_static-416x640.py']

codebase_config = dict(
    type='mmdet3d',
    task='MonocularDetection',
    model_type='end2end',
    ann_file='tests/test_codebase/test_mmdet3d/data/adam/img_info.json'
)

backend_config = dict(common_config=dict(fp16_mode=True, max_workspace_size=1 << 30))
