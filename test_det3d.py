import torch
import cv2

from mmengine.structures import InstanceData
from mmdet3d.structures import Det3DDataSample
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config

deploy_cfg = './configs/mmdet3d/monocular-detection/fcos3d_adam_tensorrt_static-416x640.py'
model_cfg = '/home/alan_khang/dev/mmdetection3d/work_dirs/fcos3d_adam/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_adam-mono3d.py'
device = 'cuda:0'
backend_model = ['work_dir/mmdet3d/fcos3d/end2end.engine']
image_path = '/home/alan_khang/Downloads/guilder_dataset/images/frame_000000.jpg'
image = cv2.imread(image_path)

assert image is not None, f"Image not found at {image_path}"

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)[0]

if isinstance(result, dict): 
    metadata = model_inputs['data_samples'][0].metainfo
    pred_instances_3d = InstanceData(metainfo=metadata)
    pred_instances_3d.bboxes_3d = result['img_bbox']['bboxes_3d']
    pred_instances_3d.labels_3d = result['img_bbox']['labels_3d']
    pred_instances_3d.scores_3d= result['img_bbox']['scores_3d']
    result = Det3DDataSample(metainfo=metadata)
    result.pred_instances_3d = pred_instances_3d

# visualize results
task_processor.visualize(
    image=image,
    result=result,
    window_name='visualize',
    output_file='output_detection.png',
    show=True)
