# -*- coding: utf-8 -*-
import logging
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec, batched_nms, cat, paste_masks_in_image
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import sigmoid_focal_loss_jit

from .utils import imrescale, center_of_mass, point_nms, mask_nms, matrix_nms
from .loss import dice_loss, FocalLoss

__all__ = ["SOLOv2"]


@META_ARCH_REGISTRY.register()
class SOLOv2(nn.Module):
    """
    SOLOv2 model. Creates FPN backbone, instance branch for kernels and categories prediction,
    mask branch for unified mask features.
    Calculates and applies proper losses to class and masks.
    """

    def __init__(self, cfg):
        super().__init__()

        # get the device of the model
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.scale_ranges = cfg.MODEL.SOLOV2.FPN_SCALE_RANGES
        self.strides = cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES
        self.sigma = cfg.MODEL.SOLOV2.SIGMA
        # Instance parameters.
        self.num_classes = cfg.MODEL.SOLOV2.NUM_CLASSES
        self.num_kernels = cfg.MODEL.SOLOV2.NUM_KERNELS
        self.num_grids = cfg.MODEL.SOLOV2.NUM_GRIDS

        self.instance_in_features = cfg.MODEL.SOLOV2.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS  # = fpn.
        self.instance_channels = cfg.MODEL.SOLOV2.INSTANCE_CHANNELS

        # Mask parameters.
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_in_features = cfg.MODEL.SOLOV2.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.SOLOV2.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.SOLOV2.MASK_CHANNELS
        self.num_masks = cfg.MODEL.SOLOV2.NUM_MASKS

        # Inference parameters.
        self.max_before_nms = cfg.MODEL.SOLOV2.NMS_PRE
        self.score_threshold = cfg.MODEL.SOLOV2.SCORE_THR
        self.update_threshold = cfg.MODEL.SOLOV2.UPDATE_THR
        self.mask_threshold = cfg.MODEL.SOLOV2.MASK_THR
        self.max_per_img = cfg.MODEL.SOLOV2.MAX_PER_IMG
        self.nms_kernel = cfg.MODEL.SOLOV2.NMS_KERNEL
        self.nms_sigma = cfg.MODEL.SOLOV2.NMS_SIGMA
        self.nms_type = cfg.MODEL.SOLOV2.NMS_TYPE

        # build the backbone.
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()

        # build the ins head.
        instance_shapes = [backbone_shape[f] for f in self.instance_in_features]
        self.ins_head = SOLOv2InsHead(cfg, instance_shapes)

        # build the mask head.
        mask_shapes = [backbone_shape[f] for f in self.mask_in_features]
        self.mask_head = SOLOv2MaskHead(cfg, mask_shapes)

        # loss
        self.ins_loss_weight = cfg.MODEL.SOLOV2.LOSS.DICE_WEIGHT
        self.focal_loss_alpha = cfg.MODEL.SOLOV2.LOSS.FOCAL_ALPHA
        self.focal_loss_gamma = cfg.MODEL.SOLOV2.LOSS.FOCAL_GAMMA
        self.focal_loss_weight = cfg.MODEL.SOLOV2.LOSS.FOCAL_WEIGHT

        # image transform
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.
            For now, each item in the list is a dict that contains:
                image: Tensor, image in (C, H, W) format.
                instances: Instances
                Other information that's included in the original dicts, such as:
                    "height", "width" (int): the output resolution of the model, used in inference.
                        See :meth:`postprocess` for details.
         Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        if not isinstance(batched_inputs, list):
            batched_inputs = [batched_inputs]
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "Targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'Targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["Targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        # ins branch
        ins_features = [features[f] for f in self.instance_in_features]
        ins_features = self.split_feats(ins_features)
        cate_pred, kernel_pred = self.ins_head(ins_features)

        # mask branch
        mask_features = [features[f] for f in self.mask_in_features]
        mask_pred = self.mask_head(mask_features)

        if self.training:
            """
            get_ground_truth.
            return loss and so on.
            """
            mask_feat_size = mask_pred.size()[-2:]
            Targets = self.get_ground_truth(gt_instances, mask_feat_size)
            losses = self.loss(cate_pred, kernel_pred, mask_pred, Targets)
            return losses
        else:
            # point nms.
            cate_pred = [point_nms(cate_p.sigmoid(), kernel=2).permute(0, 2, 3, 1)
                         for cate_p in cate_pred]
            # do inference for results.
            results = self.inference(cate_pred, kernel_pred, mask_pred, images.image_sizes, batched_inputs)
            return results


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @torch.no_grad()
    def get_ground_truth(self, gt_instances, mask_feat_size=None):
        Instance_GT_List, Category_GT_List, Instance_Indication_GT_list, grid_order_list = [], [], [], []
        for img_idx in range(len(gt_instances)):
            cur_Instance_GT_List, cur_Category_GT_List, \
            cur_Instance_Indication_GT_list, cur_grid_order_list = \
                self.get_ground_truth_single(img_idx, gt_instances,
                                             mask_feat_size=mask_feat_size)
            Instance_GT_List.append(cur_Instance_GT_List)
            Category_GT_List.append(cur_Category_GT_List)
            Instance_Indication_GT_list.append(cur_Instance_Indication_GT_list)
            grid_order_list.append(cur_grid_order_list)
        return Instance_GT_List, Category_GT_List, Instance_Indication_GT_list, grid_order_list
        
    def get_ground_truth_single(self, img_idx, gt_instances, mask_feat_size):
        gt_bboxes_raw = gt_instances[img_idx].gt_boxes.tensor
        GT_Classes_raw = gt_instances[img_idx].gt_classes
        gt_masks_raw = gt_instances[img_idx].gt_masks.tensor
        device = GT_Classes_raw[0].device

        # ins
        gt_areas = torch.sqrt((gt_bboxes_raw[:, 2] - gt_bboxes_raw[:, 0]) * (
                gt_bboxes_raw[:, 3] - gt_bboxes_raw[:, 1]))

        Instance_GT_List = []
        Category_GT_List = []
        Instance_Indication_GT_list = []
        grid_order_list = []
        for (lower_bound, upper_bound), stride, num_grid \
                in zip(self.scale_ranges, self.strides, self.num_grids):

            # Find where are the instances that satisfy the area conditions
            hit_indices = ((gt_areas >= lower_bound) & (gt_areas <= upper_bound)).nonzero().flatten()
            num_ins = len(hit_indices)

            Instance_GT = []
            grid_order = []
            # Create a SxS grid to store GT categories
            cate_label = torch.zeros([num_grid, num_grid], dtype=torch.int64, device=device)
            cate_label = torch.fill_(cate_label, self.num_classes)
            # Create an array with S^2 elements to indicate ... (gridde instance olup olmadığını tutar)
            Instance_Indication_GT = torch.zeros([num_grid ** 2], dtype=torch.bool, device=device)
            
            # No instances with specifies area size
            if num_ins == 0:
                Instance_GT = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
                Instance_GT_List.append(Instance_GT)
                Category_GT_List.append(cate_label)
                Instance_Indication_GT_list.append(Instance_Indication_GT)
                grid_order_list.append([])
                continue

            # Get GTs for hit indices
            gt_bboxes = gt_bboxes_raw[hit_indices]
            GT_Classes = GT_Classes_raw[hit_indices]
            gt_masks = gt_masks_raw[hit_indices, ...]

            half_ws = 0.5 * (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * self.sigma
            half_hs = 0.5 * (gt_bboxes[:, 3] - gt_bboxes[:, 1]) * self.sigma

            # Mass center
            center_ws, center_hs = center_of_mass(gt_masks)
            valid_mask_flags = gt_masks.sum(dim=-1).sum(dim=-1) > 0

            #########################################
            # Some numerical shit starting
            output_stride = 4
            gt_masks = gt_masks.permute(1, 2, 0).to(dtype=torch.uint8).cpu().numpy()
            gt_masks = imrescale(gt_masks, scale=1./output_stride)
            if len(gt_masks.shape) == 2:
                gt_masks = gt_masks[..., None]
            gt_masks = torch.from_numpy(gt_masks).to(dtype=torch.uint8, device=device).permute(2, 0, 1)
            # Ended
            #########################################

            for seg_mask, gt_label, half_h, half_w, center_h, center_w, valid_mask_flag in\
                zip(gt_masks, GT_Classes, half_hs, half_ws, center_hs, center_ws, valid_mask_flags):

                if not valid_mask_flag:
                    continue

                # sizes
                upsampled_size = (mask_feat_size[0] * 4, mask_feat_size[1] * 4)
                coord_w = int((center_w / upsampled_size[1]) // (1. / num_grid))
                coord_h = int((center_h / upsampled_size[0]) // (1. / num_grid))

                # left, top, right, down
                # finds where these boxes in grids
                top_box = max(0, int(((center_h - half_h) / upsampled_size[0]) // (1. / num_grid)))
                down_box = min(num_grid - 1, int(((center_h + half_h) / upsampled_size[0]) // (1. / num_grid)))
                left_box = max(0, int(((center_w - half_w) / upsampled_size[1]) // (1. / num_grid)))
                right_box = min(num_grid - 1, int(((center_w + half_w) / upsampled_size[1]) // (1. / num_grid)))

                # grid coords
                top = max(top_box, coord_h-1)
                down = min(down_box, coord_h+1)
                left = max(coord_w-1, left_box)
                right = min(right_box, coord_w+1)

                # Category assignments
                cate_label[top:(down+1), left:(right+1)] = gt_label

                # Maskenin hangi gridlere denk geldiğini bulup oralara gerekli atamaları yap
                for i in range(top, down+1):
                    for j in range(left, right+1):
                        # Gridin i * S + j şeklinde flatten halinin indexi
                        Flat_Grid_Index = int(i * num_grid + j)

                        # mask_feat_size is the prediction size therefore the maximum mask size
                        Current_Instance_GT = torch.zeros([mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8,
                                                    device=device)
                        # Put mask
                        Current_Instance_GT[:seg_mask.shape[0], :seg_mask.shape[1]] = seg_mask
                        Instance_GT.append(Current_Instance_GT)
                        Instance_Indication_GT[Flat_Grid_Index] = True
                        # Grid order tut ki hangisi hangisi bilesin
                        grid_order.append(Flat_Grid_Index)
            if len(Instance_GT) == 0:
                Instance_GT = torch.zeros([0, mask_feat_size[0], mask_feat_size[1]], dtype=torch.uint8, device=device)
            else:
                Instance_GT = torch.stack(Instance_GT, 0)
            Instance_GT_List.append(Instance_GT)
            Category_GT_List.append(cate_label)
            Instance_Indication_GT_list.append(Instance_Indication_GT)
            grid_order_list.append(grid_order)
        return Instance_GT_List, Category_GT_List, Instance_Indication_GT_list, grid_order_list

    def loss(self, Category_Predictions, Kernel_Predictions, Instance_Predictions, Targets):
        Instance_GT_List, Category_GT_List, Instance_Indication_GT_list, grid_order_list = Targets
        # ins
        Instance_GTs = [torch.cat([Instance_GTs_level_img
                                 for Instance_GTs_level_img in Instance_GTs_level], 0)
                      for Instance_GTs_level in zip(*Instance_GT_List)]

        # Kernel_Predictions = [[Kernel_Predictions_level_img.view(Kernel_Predictions_level_img.shape[0], -1)[:, grid_orders_level_img]
        #                  for Kernel_Predictions_level_img, grid_orders_level_img in
        #                  zip(Kernel_Predictions_level, grid_orders_level)]
        #                 for Kernel_Predictions_level, grid_orders_level in zip(Kernel_Predictions, zip(*grid_order_list))]
        # Equivalent of ^^^^^^^^^^^^^ is given vvvvvvvvvvvvvvvvvvvvv
        Kernel_Predictions = []
        for Kernel_Predictions_level, grid_orders_level in zip(Kernel_Predictions, zip(*grid_order_list)):
            Kernel_Predictions_level_img_list = []
            for Kernel_Predictions_level_img, grid_order_level_img in zip(Kernel_Predictions_level, grid_orders_level):
                Kernel_Predictions_level_img_view = Kernel_Predictions_level_img.view(Kernel_Predictions_level_img.shape[0], -1)
                Kernel_Predictions_level_img_selected = Kernel_Predictions_level_img_view[:, grid_order_level_img]
                Kernel_Predictions_level_img_list.append(Kernel_Predictions_level_img_selected)
            Kernel_Predictions.append(Kernel_Predictions_level_img_list)

        # generate masks
        Instance_Predictions_list = []
        for b_kernel_pred in Kernel_Predictions:
            b_mask_pred = []
            for idx, kernel_pred in enumerate(b_kernel_pred):

                if kernel_pred.size()[-1] == 0:
                    continue
                cur_Instance_Predictions = Instance_Predictions[idx, ...]
                H, W = cur_Instance_Predictions.shape[-2:]
                N, I = kernel_pred.shape
                cur_Instance_Predictions = cur_Instance_Predictions.unsqueeze(0)
                kernel_pred = kernel_pred.permute(1, 0).view(I, -1, 1, 1)
                cur_Instance_Predictions = F.conv2d(cur_Instance_Predictions, kernel_pred, stride=1).view(-1, H, W)
                b_mask_pred.append(cur_Instance_Predictions)
            if len(b_mask_pred) == 0:
                b_mask_pred = None
            else:
                b_mask_pred = torch.cat(b_mask_pred, 0)
            Instance_Predictions_list.append(b_mask_pred)

        Instance_Indication_GTs = [
            torch.cat([Instance_Indication_GTs_level_img.flatten()
                       for Instance_Indication_GTs_level_img in Instance_Indication_GTs_level])
            for Instance_Indication_GTs_level in zip(*Instance_Indication_GT_list)
        ]
        flatten_Instance_Indication_GTs = torch.cat(Instance_Indication_GTs)

        num_ins = flatten_Instance_Indication_GTs.sum()

        # dice loss
        loss_ins = []
        for input, target in zip(Instance_Predictions_list, Instance_GTs):
            if input is None:
                continue
            input = torch.sigmoid(input)
            loss_ins.append(dice_loss(input, target))

        loss_ins_mean = torch.cat(loss_ins).mean()
        loss_ins = loss_ins_mean * self.ins_loss_weight

        # cate
        cate_labels = [
            torch.cat([cate_labels_level_img.flatten()
                       for cate_labels_level_img in cate_labels_level])
            for cate_labels_level in zip(*Category_GT_List)
        ]
        flatten_cate_labels = torch.cat(cate_labels)

        Category_Predictions = [
            cate_pred.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cate_pred in Category_Predictions
        ]
        flatten_Category_Predictions = torch.cat(Category_Predictions)

        # prepare one_hot
        pos_inds = torch.nonzero(flatten_cate_labels != self.num_classes).squeeze(1)

        flatten_cate_labels_oh = torch.zeros_like(flatten_Category_Predictions)
        flatten_cate_labels_oh[pos_inds, flatten_cate_labels[pos_inds]] = 1

        loss_cate = self.focal_loss_weight * sigmoid_focal_loss_jit(flatten_Category_Predictions, flatten_cate_labels_oh,
                                    gamma=self.focal_loss_gamma,
                                    alpha=self.focal_loss_alpha,
                                    reduction="sum") / (num_ins + 1)
        return {'loss_ins': loss_ins,
                'loss_cate': loss_cate}
    
    '''
    import torch
    from torch.nn import functional as F


    def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
        reduction: str = "none",
    ) -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss


    sigmoid_focal_loss_jit: "torch.jit.ScriptModule" = torch.jit.script(sigmoid_focal_loss)


    def sigmoid_focal_loss_star(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 1,
        reduction: str = "none",
    ) -> torch.Tensor:
        """
        FL* described in RetinaNet paper Appendix: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Gamma parameter described in FL*. Default = 1 (no weighting).
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        inputs = inputs.float()
        targets = targets.float()
        shifted_inputs = gamma * (inputs * (2 * targets - 1))
        loss = -(F.logsigmoid(shifted_inputs)) / gamma

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss *= alpha_t

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss


    sigmoid_focal_loss_star_jit: "torch.jit.ScriptModule" = torch.jit.script(
        sigmoid_focal_loss_star
    )
    '''

    @staticmethod
    def split_feats(feats):
        return (F.interpolate(feats[0], scale_factor=0.5, mode='bilinear'),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4], size=feats[3].shape[-2:], mode='bilinear'))


    def inference(self, pred_cates, pred_kernels, pred_masks, cur_sizes, images):
        assert len(pred_cates) == len(pred_kernels)

        results = []
        num_ins_levels = len(pred_cates)
        for img_idx in range(len(images)):
            # image size.
            ori_img = images[img_idx]
            height, width = ori_img["height"], ori_img["width"]
            ori_size = (height, width)

            # prediction.
            pred_cate = [pred_cates[i][img_idx].view(-1, self.num_classes).detach()
                          for i in range(num_ins_levels)]
            pred_kernel = [pred_kernels[i][img_idx].permute(1, 2, 0).view(-1, self.num_kernels).detach()
                            for i in range(num_ins_levels)]
            pred_mask = pred_masks[img_idx, ...].unsqueeze(0)

            pred_cate = torch.cat(pred_cate, dim=0)
            pred_kernel = torch.cat(pred_kernel, dim=0)

            # inference for single image.
            result = self.inference_single_image(pred_cate, pred_kernel, pred_mask,
                                                 cur_sizes[img_idx], ori_size)
            results.append({"instances": result})
        return results

    def inference_single_image(
            self, Category_Predictions, Kernel_Predictions, seg_preds, cur_size, ori_size
    ):
        # overall info.
        h, w = cur_size
        f_h, f_w = seg_preds.size()[-2:]
        ratio = math.ceil(h/f_h)
        upsampled_size_out = (int(f_h*ratio), int(f_w*ratio))

        # process.
        inds = (Category_Predictions > self.score_threshold)
        cate_scores = Category_Predictions[inds]
        if len(cate_scores) == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        # cate_labels & Kernel_Predictions
        inds = inds.nonzero()
        cate_labels = inds[:, 1]
        Kernel_Predictions = Kernel_Predictions[inds[:, 0]]

        # trans vector.
        size_trans = cate_labels.new_tensor(self.num_grids).pow(2).cumsum(0)
        strides = Kernel_Predictions.new_ones(size_trans[-1])

        n_stage = len(self.num_grids)
        strides[:size_trans[0]] *= self.instance_strides[0]
        for ind_ in range(1, n_stage):
            strides[size_trans[ind_ - 1]:size_trans[ind_]] *= self.instance_strides[ind_]
        strides = strides[inds[:, 0]]


        # mask encoding.
        N, I = Kernel_Predictions.shape
        Kernel_Predictions = Kernel_Predictions.view(N, I, 1, 1)
        seg_preds = F.conv2d(seg_preds, Kernel_Predictions, stride=1).squeeze(0).sigmoid()

        # mask.
        seg_masks = seg_preds > self.mask_threshold
        sum_masks = seg_masks.sum((1, 2)).float()

        # filter.
        keep = sum_masks > strides
        if keep.sum() == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        seg_masks = seg_masks[keep, ...]
        seg_preds = seg_preds[keep, ...]
        sum_masks = sum_masks[keep]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # maskness.
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cate_scores *= seg_scores

        # sort and keep top nms_pre
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_before_nms:
            sort_inds = sort_inds[:self.max_before_nms]
        seg_masks = seg_masks[sort_inds, :, :]
        seg_preds = seg_preds[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        if self.nms_type == "matrix":
            # matrix nms & filter.
            cate_scores = matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores,
                                          sigma=self.nms_sigma, kernel=self.nms_kernel)
            keep = cate_scores >= self.update_threshold
        elif self.nms_type == "mask":
            # original mask nms.
            keep = mask_nms(cate_labels, seg_masks, sum_masks, cate_scores,
                                 nms_thr=self.mask_threshold)
        else:
            raise NotImplementedError

        if keep.sum() == 0:
            results = Instances(ori_size)
            results.scores = torch.tensor([])
            results.pred_classes = torch.tensor([])
            results.pred_masks = torch.tensor([])
            results.pred_boxes = Boxes(torch.tensor([]))
            return results

        seg_preds = seg_preds[keep, :, :]
        cate_scores = cate_scores[keep]
        cate_labels = cate_labels[keep]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > self.max_per_img:
            sort_inds = sort_inds[:self.max_per_img]
        seg_preds = seg_preds[sort_inds, :, :]
        cate_scores = cate_scores[sort_inds]
        cate_labels = cate_labels[sort_inds]

        # reshape to original size.
        seg_preds = F.interpolate(seg_preds.unsqueeze(0),
                                  size=upsampled_size_out,
                                  mode='bilinear')[:, :, :h, :w]
        seg_masks = F.interpolate(seg_preds,
                                  size=ori_size,
                                  mode='bilinear').squeeze(0)
        seg_masks = seg_masks > self.mask_threshold

        results = Instances(ori_size)
        results.pred_classes = cate_labels
        results.scores = cate_scores
        results.pred_masks = seg_masks

        # get bbox from mask
        pred_boxes = torch.zeros(seg_masks.size(0), 4)
        for i in range(seg_masks.size(0)):
           mask = seg_masks[i].squeeze()
           ys, xs = torch.where(mask)
           pred_boxes[i] = torch.tensor([xs.min(), ys.min(), xs.max(), ys.max()]).float()
        results.pred_boxes = Boxes(pred_boxes)

        return results


class SOLOv2InsHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        SOLOv2 Instance Head. (category head)
        """
        super().__init__()
        # fmt: off
        self.num_classes = cfg.MODEL.SOLOV2.NUM_CLASSES
        self.num_kernels = cfg.MODEL.SOLOV2.NUM_KERNELS
        self.num_grids = cfg.MODEL.SOLOV2.NUM_GRIDS
        self.instance_in_features = cfg.MODEL.SOLOV2.INSTANCE_IN_FEATURES
        self.instance_strides = cfg.MODEL.SOLOV2.FPN_INSTANCE_STRIDES
        self.instance_in_channels = cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS  # = fpn.
        self.instance_channels = cfg.MODEL.SOLOV2.INSTANCE_CHANNELS
        # Convolutions to use in the towers
        self.type_dcn = cfg.MODEL.SOLOV2.TYPE_DCN
        self.num_levels = len(self.instance_in_features)
        assert self.num_levels == len(self.instance_strides), \
            print("Strides should match the features.")
        # fmt: on

        head_configs = {"cate": (cfg.MODEL.SOLOV2.NUM_INSTANCE_CONVS,
                                 cfg.MODEL.SOLOV2.USE_DCN_IN_INSTANCE,
                                 False),
                        "kernel": (cfg.MODEL.SOLOV2.NUM_INSTANCE_CONVS,
                                   cfg.MODEL.SOLOV2.USE_DCN_IN_INSTANCE,
                                   cfg.MODEL.SOLOV2.USE_COORD_CONV)
                        }

        norm = None if cfg.MODEL.SOLOV2.NORM == "none" else cfg.MODEL.SOLOV2.NORM
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, \
            print("Each level must have the same channel!")
        in_channels = in_channels[0]
        assert in_channels == cfg.MODEL.SOLOV2.INSTANCE_IN_CHANNELS, \
            print("In channels should equal to tower in channels!")

        for head in head_configs:
            tower = []
            num_convs, use_deformable, use_coord = head_configs[head]
            for i in range(num_convs):
                conv_func = nn.Conv2d
                if i == 0:
                    if use_coord:
                        chn = self.instance_in_channels + 2
                    else:
                        chn = self.instance_in_channels
                else:
                    chn = self.instance_channels

                tower.append(conv_func(
                        chn, self.instance_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                ))
                if norm == "GN":
                    tower.append(nn.GroupNorm(32, self.instance_channels))
                tower.append(nn.ReLU(inplace=True))
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        self.cate_pred = nn.Conv2d(
            self.instance_channels, self.num_classes,
            kernel_size=3, stride=1, padding=1
        )
        self.kernel_pred = nn.Conv2d(
            self.instance_channels, self.num_kernels,
            kernel_size=3, stride=1, padding=1
        )

        for modules in [
            self.cate_tower, self.kernel_tower,
            self.cate_pred, self.kernel_pred,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.SOLOV2.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cate_pred.bias, bias_value)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pass
        """
        cate_pred = []
        kernel_pred = []

        for idx, feature in enumerate(features):
            ins_kernel_feat = feature
            print(f"category head || {idx}th feature size: {ins_kernel_feat.shape}")
            # concat coord
            x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
            y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
            y, x = torch.meshgrid(y_range, x_range)
            y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
            coord_feat = torch.cat([x, y], 1)
            ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

            # individual feature.
            kernel_feat = ins_kernel_feat
            seg_num_grid = self.num_grids[idx]
            kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear')
            print(f"Kernel features size: {kernel_feat.shape}")
            cate_feat = kernel_feat[:, :-2, :, :]

            # kernel
            kernel_feat = self.kernel_tower(kernel_feat)
            kernel_pred.append(self.kernel_pred(kernel_feat))

            # cate
            cate_feat = self.cate_tower(cate_feat)
            cate_pred.append(self.cate_pred(cate_feat))

        print("predictions of category head: ")
        for i, (c, k) in enumerate(zip(cate_pred, kernel_pred)):
            print(f"{i}th cate pred shape: {c.shape}")
            print(f"{i}th kernel pred shape: {k.shape}")
            print(f"{i}th grid size: {self.num_grids[i]}")
        return cate_pred, kernel_pred


class SOLOv2MaskHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        SOLOv2 Mask Head.
        """
        super().__init__()
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        self.num_masks = cfg.MODEL.SOLOV2.NUM_MASKS
        self.mask_in_features = cfg.MODEL.SOLOV2.MASK_IN_FEATURES
        self.mask_in_channels = cfg.MODEL.SOLOV2.MASK_IN_CHANNELS
        self.mask_channels = cfg.MODEL.SOLOV2.MASK_CHANNELS
        self.num_levels = len(input_shape)
        assert self.num_levels == len(self.mask_in_features), \
            print("Input shape should match the features.")
        # fmt: on
        norm = None if cfg.MODEL.SOLOV2.NORM == "none" else cfg.MODEL.SOLOV2.NORM

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.num_levels):
            convs_per_level = nn.Sequential()
            if i == 0:
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_in_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(i), nn.Sequential(*conv_tower))
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    chn = self.mask_in_channels + 2 if i == 3 else self.mask_in_channels
                    conv_tower = list()
                    conv_tower.append(nn.Conv2d(
                        chn, self.mask_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=norm is None
                    ))
                    if norm == "GN":
                        conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                    conv_tower.append(nn.ReLU(inplace=False))
                    convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                    upsample_tower = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module(
                        'upsample' + str(j), upsample_tower)
                    continue
                conv_tower = list()
                conv_tower.append(nn.Conv2d(
                    self.mask_channels, self.mask_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
                ))
                if norm == "GN":
                    conv_tower.append(nn.GroupNorm(32, self.mask_channels))
                conv_tower.append(nn.ReLU(inplace=False))
                convs_per_level.add_module('conv' + str(j), nn.Sequential(*conv_tower))
                upsample_tower = nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=False)
                convs_per_level.add_module('upsample' + str(j), upsample_tower)

            self.convs_all_levels.append(convs_per_level)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                self.mask_channels, self.num_masks,
                kernel_size=1, stride=1,
                padding=0, bias=norm is None),
            nn.GroupNorm(32, self.num_masks),
            nn.ReLU(inplace=True)
        )

        for modules in [self.convs_all_levels, self.conv_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pass
        """
        assert len(features) == self.num_levels, \
            print("The number of input features should be equal to the supposed level.")

        # bottom features first.
        feature_add_all_level = self.convs_all_levels[0](features[0])
        for i in range(1, self.num_levels):
            mask_feat = features[i]
            print(f"{i}th mask_feat shape: {mask_feat.shape}")
            if i == 3:  # add for coord.
                x_range = torch.linspace(-1, 1, mask_feat.shape[-1], device=mask_feat.device)
                y_range = torch.linspace(-1, 1, mask_feat.shape[-2], device=mask_feat.device)
                y, x = torch.meshgrid(y_range, x_range)
                y = y.expand([mask_feat.shape[0], 1, -1, -1])
                x = x.expand([mask_feat.shape[0], 1, -1, -1])
                coord_feat = torch.cat([x, y], 1)
                mask_feat = torch.cat([mask_feat, coord_feat], 1)
            # add for top features.
            feature_add_all_level = feature_add_all_level + self.convs_all_levels[i](mask_feat)

        mask_pred = self.conv_pred(feature_add_all_level)
        print(f"mask_pred shape: {mask_pred.shape}")
        return mask_pred
