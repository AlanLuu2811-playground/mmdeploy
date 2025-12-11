# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmdet.utils import ConfigType
from torch import Tensor

from mmdeploy.core import FUNCTION_REWRITER

@FUNCTION_REWRITER.register_rewriter(
    'src.roi_heads.standard_roi_head.StandardRoIHead.predict_mask')
def standard_roi_head__predict_mask(self,
                                    x: Tuple[Tensor],
                                    batch_img_metas: List[dict],
                                    results_list: List[Tensor],
                                    rescale: bool = False) -> List[Tensor]:
    """Perform forward propagation of the mask head and predict detection
    results on the features of the upstream network.

    Args:
        x (tuple[Tensor]): Feature maps of all scale level.
        batch_img_metas (list[dict]): List of image information.
        results_list (list[:obj:`InstanceData`]): Detection results of
            each image.
        rescale (bool): If True, return boxes in original image space.
            Defaults to False.

    Returns:
        list[Tensor]: Detection results of each image
        after the post process.
        Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
                (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
                the last dimension 4 arrange as (x1, y1, x2, y2).
            - masks (Tensor): Has a shape (num_instances, H, W).
    """
    dets, det_labels = results_list
    batch_size = dets.size(0)
    det_bboxes = dets[..., :4]
    # expand might lead to static shape, use broadcast instead
    batch_index = torch.arange(
        det_bboxes.size(0), device=det_bboxes.device).float().view(
            -1, 1, 1) + det_bboxes.new_zeros(
                (det_bboxes.size(0), det_bboxes.size(1))).unsqueeze(-1)
    mask_rois = torch.cat([batch_index, det_bboxes], dim=-1)
    mask_rois = mask_rois.view(-1, 5)
    mask_results = self._mask_forward(x, mask_rois)
    mask_preds = mask_results['mask_preds']
    num_det = det_bboxes.shape[1]
    segm_results = self.mask_head.predict_by_feat(
        mask_preds,
        results_list,
        batch_img_metas,
        self.test_cfg,
        rescale=rescale)
    segm_results = segm_results.reshape(batch_size, num_det,
                                        segm_results.shape[-2],
                                        segm_results.shape[-1])
    return dets, det_labels, segm_results
