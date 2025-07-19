# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor
import torch

from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import is_dynamic_shape


def __forward_impl(self, ctx, img, cam2img, cam2img_inverse, img_metas,
                   **kwargs):
    """Rewrite `forward` function for SingleStageMono3DDetector.

    Support both dynamic and static export to onnx.
    """
    assert isinstance(img, torch.Tensor)

    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    # get origin input shape as tensor to support onnx dynamic shape
    img_shape = torch._shape_as_tensor(img)[2:]
    if not is_dynamic_flag:
        img_shape = [int(val) for val in img_shape]
    img_metas['img_shape'] = img_shape
    return self.predict(
        img, cam2img, cam2img_inverse, img_metas, rescale=True)

#@FUNCTION_REWRITER.register_rewriter(
#    'mmdet3d.models.detectors.single_stage_mono3d.'
#    'SingleStageMono3DDetector.forward')
#def singlestagemono3ddetector__forward(self, inputs: Tensor, **kwargs):
#    """Rewrite to support feed inputs of Tensor type.
#
#    Args:
#        inputs (Tensor): Input image
#
#    Returns:
#        list: two torch.Tensor
#    """
#
#    x = self.extract_feat({'imgs': inputs})
#    results = self.bbox_head.forward(x)
#    return results[0], results[1]

@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.single_stage_mono3d.SingleStageMono3DDetector.'
    'forward')
def singlestagemono3ddetector__forward(
    self,
    img,
    cam2img,
    cam2img_inverse,
    data_samples=None,
    return_loss=False,
    **kwargs):
    """Rewrite this function to run the model directly."""
    img_metas = data_samples
    if img_metas is None:
        img_metas = [{}]
    else:
        assert len(img_metas) == 1, 'do not support aug_test'
        img_metas = img_metas[0]

    if isinstance(img, list):
        img = img[0]

    ctx = FUNCTION_REWRITER.get_context()

    return __forward_impl(
        self,
        ctx,
        img,
        cam2img,
        cam2img_inverse,
        img_metas=img_metas,
        **kwargs)
