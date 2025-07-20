import torch

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config

deploy_cfg = './configs/mmdet/detection/detection_tensorrt_dynamic-320x320-1344x1344.py'
model_cfg = '/home/alan_khang/dev/mmdetection/rtmdet_tiny_8xb32-300e_coco.py'
device = 'cuda:0'
backend_model = ['work_dir/mmdet/rtmdet/end2end.engine']
image = './demo/resources/det.jpg'

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

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result,
    window_name='visualize',
    output_file='output_detection.png',
    show=True)
