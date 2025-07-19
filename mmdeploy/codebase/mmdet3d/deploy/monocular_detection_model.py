# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Sequence, Union

import mmcv
import torch
from mmengine import Config
from mmengine.registry import Registry
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor
from mmengine.structures import BaseDataElement, InstanceData

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            load_config)

from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.structures import (CameraInstance3DBoxes, LiDARInstance3DBoxes,
                                bbox3d2result, xywhr2xyxyr)

__BACKEND_MODEL = Registry('backend_mono_detectors')

@__BACKEND_MODEL.register_module('end2end')
class MonocularDetectionModel(BaseBackendModel):
    """End to end model for inference of monocular detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string specifying device type.
        model_cfg (str | mmcv.Config): The model config.
        deploy_cfg (str| mmcv.Config): Deployment config file or loaded Config
            object.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 model_cfg: Config,
                 deploy_cfg: Union[str, Config] = None,
                 data_preprocessor: Optional[Union[dict,
                                                   torch.nn.Module]] = None,
                 **kwargs):
        super().__init__(deploy_cfg=deploy_cfg, data_preprocessor=data_preprocessor)
        self.deploy_cfg = deploy_cfg
        self.model_cfg = model_cfg
        self.device = device
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str):
        """Initialize backend wrapper.

        Args:
            backend (Backend): The backend enum, specifying backend type.
            backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            device (str): A string specifying device type.
        """
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            input_names=[self.input_name],
            output_names=output_names,
            deploy_cfg=self.deploy_cfg)
    
    def forward(self,
                inputs: dict,
                data_samples: Optional[List[BaseDataElement]] = None,
                **kwargs) -> Any:
        """Run forward inference.

        Args:
            inputs (dict): A dict contains `imgs`
            data_samples (List[BaseDataElement]): A list of meta info for
                image(s).
        Returns:
            list: A list contains predictions.
        """
        img_metas = data_samples[0].metainfo
        preprocessed = inputs['imgs']
        input_dict = {
            'input': preprocessed.to(self.device),
            'cam2img': inputs['cam2img'].to(self.device),
            'cam2img_inverse': inputs['cam2img_inverse'].to(self.device)
        }
        outputs = self.wrapper(input_dict)
        outputs = self.wrapper.output_to_list(outputs)
        outputs = [x.squeeze(0) for x in outputs]
        outputs[0] = img_metas['box_type_3d'](
            outputs[0], 
            box_dim=9, 
            origin=(0.5, 0.5, 0.5))
        outputs.pop(3)  # pop dir_scores

        bbox_img = [bbox3d2result(*outputs)]

        bbox_list = [dict() for i in range(len(bbox_img))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        return bbox_list

def build_monocular_detection_model(
    model_files: Sequence[str],
    model_cfg: Union[str, Config],
    deploy_cfg: Union[str, Config],
    device: str,
    data_preprocessor: Optional[Union[Config, BaseDataPreprocessor]] = None,
    **kwargs):
    """Build monocular detection model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model
    Returns:
        VMonocularDetectionModel: Detector for a configured backend.
    """
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')

    backend_detector = __BACKEND_MODEL.build(
        dict(
            type=model_type,
            backend=backend,
            backend_files=model_files,
            device=device,
            model_cfg=model_cfg,
            deploy_cfg=deploy_cfg,
            data_preprocessor=data_preprocessor,
            **kwargs))

    return backend_detector
