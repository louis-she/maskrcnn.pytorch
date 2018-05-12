import functools

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from nms.nms_wrapper import nms
from utils import apply_box_deltas, clip_boxes, get_iou, log2, generate_pyramid_anchors
from roialign.roi_align.crop_and_resize import CropAndResizeFunction

device = torch.device("cuda")

"""
ResNet50
"""
class Downsample(nn.Module):
    """This net is used to change the
        channel and output shape of resnet
    """

    def __init__(self, in_planes, out_planes, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return x

class Bottleneck(nn.Module):

    def __init__(self, in_planes, planes, downsample=False, stride=1):
        """Bottleneck structure for ResNet
        Channel changes:
            1. inplanes -> planes
            2. planes -> planes
            3. planes -> 4 * planes
        Shape changes:
            if downsample is true then
                1. aaa x bbb -> aaa x bbb
                2. aaa x bbb -> aaa / 2 x bbb / 2
                3. aaa/2 x bbb/2 -> aaa/2 x bbb/2
            else:
                stays the same
        """
        super().__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if downsample:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.downsample = downsample
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample:
            residual = self.downsample(residual)

        x = x + residual
        x = self.relu(x)
        return x

class ResNet50(nn.Module):
    """Architecture refer to: https://wiseodd.github.io/techblog/2016/10/13/residual-net/
       or torchvision.models.resnet50()
    """

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            # In most cases, padding set to `kernel_size // 2` to make sure a correct output size
            # Refer to the fomular at http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), # 112 x 112 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = self.__create_bottlenecks(64, 64, 3, stride=1, downsample=Downsample(64, 256, 1))
        self.layer3 = self.__create_bottlenecks(256, 128, 4, downsample=Downsample(256, 512, 2))
        self.layer4 = self.__create_bottlenecks(512, 256, 6, downsample=Downsample(512, 1024, 2))
        self.layer5 = self.__create_bottlenecks(1024, 512, 3, downsample=Downsample(1024, 2048, 2))
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def __create_bottlenecks(self, in_planes, planes, num_blocks, stride=2, downsample=False):
        bottlenecks = []

        bottlenecks.append(Bottleneck(in_planes, planes, stride=stride, downsample=downsample))
        for i in range(1, num_blocks):
            bottlenecks.append(Bottleneck(planes*4, planes, stride=stride))
        return nn.Sequential(*bottlenecks)

    def layers(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        return x


class FPN(nn.Module):
    """Refer to https://www.jiqizhixin.com/articles/2017-07-25-2
    """

    def __init__(self, p1, p2, p3, p4, p5, out_planes):
        """
        out_planes
        """
        super().__init__()
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p5 = p5

        self.p5_conv1 = nn.Conv2d(2048, out_planes, kernel_size=1, stride=1)
        self.p5_conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)

        self.p4_conv1 = nn.Conv2d(1024, out_planes, kernel_size=1, stride=1)
        self.p4_conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)

        self.p3_conv1 = nn.Conv2d(512, out_planes, kernel_size=1, stride=1)
        self.p3_conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)

        self.p2_conv1 = nn.Conv2d(256, out_planes, kernel_size=1, stride=1)
        self.p2_conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        p1_out = self.p1(x) # 64 x 56 x 56
        p2_out = self.p2(p1_out) # 256 x 56 x 56
        p3_out = self.p3(p2_out) # 512 x 28 x 28
        p4_out = self.p4(p3_out) # 1024 x 14 x 14
        p5_out = self.p5(p4_out) # 2048 x 7 x 7

        # change channel
        p5_out = self.p5_conv1(p5_out)
        p4_out = self.p4_conv1(p4_out) + torch.nn.functional.upsample(p5_out, scale_factor=2)
        p3_out = self.p3_conv1(p3_out) + torch.nn.functional.upsample(p4_out, scale_factor=2)
        p2_out = self.p2_conv1(p2_out) + torch.nn.functional.upsample(p3_out, scale_factor=2)

        p5_out = self.p5_conv2(p5_out)
        p4_out = self.p4_conv2(p4_out)
        p3_out = self.p3_conv2(p3_out)
        p2_out = self.p2_conv2(p2_out)

        return p2_out, p3_out, p4_out, p5_out


class RPN(nn.Module):
    """A very mini network produce ROIs, there are all 3 conv2d layers.
    """

    def __init__(self, in_channel, out_channel, anchors_per_loc=3, anchor_stride=1):
        super().__init__()
        self.conv_base = nn.Conv2d(in_channel, out_channel, stride=anchor_stride, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.conv_bbox = nn.Conv2d(out_channel, 4*anchors_per_loc, stride=1, kernel_size=1)
        self.conv_klass = nn.Conv2d(out_channel, 2*anchors_per_loc, stride=1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Args:
            x: some feature map from FPN
        Returns:
            bbox_output: [batch_size, -1, 4], anchors of the feature map
            class_logits: [batch_size, -1, 2] background & object for loss computing
            softmax: [batch_size, -1, 2] score of background and object
        """
        base_output = self.conv_base(x)
        base_output = self.bn(base_output)
        base_output = self.relu(base_output)

        bbox_output = self.conv_bbox(base_output)
        class_logits = self.conv_klass(base_output)

        # change class_logits' shape as [batch_size, -1, 2]
        # about contiguous, see https://www.zhihu.com/question/60321866
        class_logits = class_logits.permute(0, 2, 3, 1).contiguous().view(class_logits.size()[0], -1, 2)
        softmax = self.softmax(class_logits)

        # change bbox_output's shape as [batch_size, -1, 4]
        bbox_output = bbox_output.permute(0, 2, 3, 1).contiguous().view(bbox_output.size()[0], -1, 4)

        return bbox_output, class_logits, softmax

class Mask(nn.Module):

    def __init__(self, num_classes, depth=256, pool_size=14, image_shape=(512, 512)):
        super().__init__()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(self.depth, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256, eps=0.001)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256, eps=0.001)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256, eps=0.001)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256, eps=0.001)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.fpn_roi_pooling = FPNRoIPooling(pool_size=pool_size)

    def forward(self, fpn_feature_maps, rois):
        x = self.fpn_roi_pooling.forward(fpn_feature_maps, rois)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)

        return x

class RegionProposal(object):

    def __init__(self, nms_threshold=0.7, proposals_num=2000, score_filter_num=6000,
            original_image_shape=[512, 512]):
        self.nms_threshold = nms_threshold
        self.proposals_num = 2000
        self.score_filter_num = score_filter_num
        self.original_image_shape = original_image_shape

    def forward(self, deltas, scores, anchors):
        """
        Args:
            fpn_rpn_outputs: outpus of RPN for every fpn features, list of 3 elements,
                             (bbox_output, class_logits, softmax)
            anchors: anchors we get from generate_pyramid_anchors
        """
        # get rid of the batch size dimension
        deltas = deltas.squeeze(0)
        scores = scores.squeeze(0)[:, 1]

        # 1. sort and keep the first 6000 anchors by score
        scores, index = scores.sort(descending=True)
        scores = scores[:self.score_filter_num]
        index = index[:self.score_filter_num]

        anchors = anchors[index.data, :]
        deltas = deltas[index.data, :]

        # 2. apply rpn_delta to anchors to get a refined bbox
        refined_anchors = apply_box_deltas(anchors, deltas)

        # 3. apply nms to get the final results
        boxes = clip_boxes(refined_anchors, self.original_image_shape)
        # Non-max suppression
        # input: [bbox_size, (y1, x1, y2, x2, score)]
        _input = torch.cat((boxes, scores.unsqueeze(1)), 1)
        keep = nms(_input, 0.7)
        keep = keep[:self.proposals_num]
        boxes = boxes[keep, :]

        # 4. normalize coordinate
        height = self.original_image_shape[0]
        width = self.original_image_shape[1]

        norm = torch.tensor( [height, width, height, width], dtype=torch.float64, device=device )
        normalized_boxes = boxes / norm

        # Add back batch dimension
        normalized_boxes = normalized_boxes.unsqueeze(0)

        return normalized_boxes

class GenerateTarget(object):

    def __init__(self, mask_shape=(28, 28), pos_neg_ratio=0.6, image_size=(512, 512)):
        """Using RoIs and GT to generate the final target
        used to computing the loss
        """
        self.pos_neg_ratio = pos_neg_ratio
        self.mask_shape = mask_shape
        self.image_size = image_size

    def forward(self, rois, gt_box, gt_mask, gt_class):
        """Noticed that the ground truth only containes positive
        we should generate many negative anchors by hand
        """
        rois = rois.squeeze(0)
        # gt_box = gt_box.squeeze(0)
        # gt_mask = gt_mask.squeeze(0)
        # gt_class = gt_class.squeeze(0)

        # 1. the rois is normalized, so will the gt_box
        gt_box[:, 2] = gt_box[:, 2] / self.image_size[1]
        gt_box[:, 0] = gt_box[:, 0] / self.image_size[1]
        gt_box[:, 1] = gt_box[:, 1] / self.image_size[0]
        gt_box[:, 3] = gt_box[:, 3] / self.image_size[0]

        # 2. enumerate rois and gt_box, and get the rois
        roi_len = rois.size()[0]
        gt_box_len = gt_box.size()[0]

        rois_repeated = rois.repeat(1, gt_box_len).view(-1, 4)
        gt_box_repeated = gt_box.repeat(roi_len, 1)

        # 3. keep the rois which has the iou > 0.5 to the
        #    gt_box, regard these as positive rois
        iou = get_iou(rois_repeated.data, gt_box_repeated)
        iou = iou.view(roi_len, gt_box_len)

        positive_indexes = (iou > 0.5).cpu()
        roi_indexes = torch.sum( positive_indexes, dim=1)
        roi_indexes = torch.nonzero(roi_indexes).squeeze(1)

        positive_rois = rois[roi_indexes, :]

        valid_iou = iou[ roi_indexes ]
        _, argmax_index = torch.max(valid_iou, dim=1)
        positive_gtbox = gt_box[argmax_index]
        positive_gtmask = gt_mask[argmax_index]
        positive_gtclass = gt_class[argmax_index]

        box_ids = torch.arange( positive_gtmask.size()[0], dtype=torch.int, device=device )
        masks = CropAndResizeFunction(self.mask_shape[0], self.mask_shape[1], 0)(positive_gtmask.unsqueeze(1).float(), positive_rois.float(), box_ids)

        masks = masks.squeeze(1)
        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = torch.round(masks)

        # 4. generate negative(background, class 0) targets
        #    noticed that mask and bbox loss will not be added
        #    for the negative ones
        negative_indexes = (iou < 0.5).cpu()
        negative_roi_indexes = torch.nonzero(torch.sum( negative_indexes, dim=1 )).squeeze(1)
        negative_rois = rois[negative_roi_indexes]

        negative_keep_num = int(positive_rois.size()[0] // self.pos_neg_ratio)

        if not negative_keep_num > negative_rois.size()[0]:
            random_negative_index = torch.randperm( negative_rois.size()[0] )
            random_negative_keep_index = random_negative_index[:negative_keep_num]
            negative_rois = negative_rois[random_negative_keep_index, :]

        negative_count = negative_rois.size()[0]
        negative_bbox = torch.zeros(negative_count, 4, device=device)
        negative_mask = torch.zeros(negative_count, self.mask_shape[0], self.mask_shape[1], device=device)
        negative_class = torch.zeros(negative_count, device=device)

        rois = torch.cat((positive_rois, negative_rois), dim=0)
        bbox = torch.cat([positive_gtbox, negative_bbox], dim=0)
        mask = torch.cat([masks.data.double(), negative_mask], dim=0)
        classes = torch.cat([positive_gtclass, negative_class], dim=0).long()

        return rois, bbox, mask, classes

class Classifier(nn.Module):

    def __init__(self, num_classes, depth=256, pool_size=(7, 7), image_shape=(512, 512)):
        super().__init__()
        self.fpn_roi_pooling = FPNRoIPooling()
        self.depth = depth
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(self.depth, 1024, kernel_size=self.pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024, eps=0.001, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)

        self.linear_class = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.linear_bbox = nn.Linear(1024, num_classes * 4)

    def forward(self, fpn_feature_maps, rois):
        x = self.fpn_roi_pooling.forward(fpn_feature_maps, rois)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = x.view(-1,1024)
        mrcnn_class_logits = self.linear_class(x)
        mrcnn_probs = self.softmax(mrcnn_class_logits)

        mrcnn_bbox = self.linear_bbox(x)
        mrcnn_bbox = mrcnn_bbox.view(mrcnn_bbox.size()[0], -1, 4)

        return [mrcnn_class_logits, mrcnn_probs, mrcnn_bbox]


class FPNRoIPooling(object):

    def __init__(self, pool_size=7, image_shape=(512, 512)):
        """
        Args:
            pool_size: the output(pooled) feature map size
            image_shape: original input image size, this is used to compute
                the Equation 1 in FPN paper
        """
        self.pool_size = pool_size
        self.image_shape = image_shape

    def forward(self, fpn_feature_maps, rpn_output):
        """apply RoI pooling to feature map by bounding boxes
        Args:
            fpn_feature_maps: outputs of FPN
            rpn_output: outpus of Proposal networks
        """
        roi_level = self.roi_level(rpn_output)
        output = []
        box_to_level = []

        for i, level in enumerate(range(2, 6)):
            feature_map = fpn_feature_maps[i]
            boxes_index_to_this_level = roi_level == level
            if torch.sum(boxes_index_to_this_level).item() == 0:
                continue
            boxes_index_to_this_level = torch.nonzero(boxes_index_to_this_level)[:, 0]
            box_to_level.append(boxes_index_to_this_level)

            boxes_to_this_level = rpn_output[boxes_index_to_this_level.data]
            ind = torch.zeros(boxes_to_this_level.size()[0], dtype=torch.int, device=device)

            feature_map = feature_map.float()
            boxes_to_this_level = boxes_to_this_level.float()

            pooled_features = CropAndResizeFunction(self.pool_size, self.pool_size, 0)(feature_map, boxes_to_this_level, ind)
            output.append(pooled_features.double())

        output = torch.cat(output, dim=0)
        box_to_level = torch.cat(box_to_level, dim=0)
        _, box_to_level = torch.sort(box_to_level)
        output = output[box_to_level]
        return output

    def roi_level(self, boxes):
        """ compute roi level by box size, the Equation 1
        from FPN paper
        """
        y1, x1, y2, x2 = boxes.chunk(4, dim=1)
        h = y2 - y1
        w = x2 - x1

        image_area = torch.tensor([self.image_shape[0]*self.image_shape[1]], dtype=torch.float64, device=device)
        roi_level = 4 + log2(torch.sqrt(h*w)/(224.0/torch.sqrt(image_area)))
        roi_level = roi_level.round().int()
        roi_level = roi_level.clamp(2,5)
        return roi_level

class GenerateRPNTargets(object):

    def __init__(self, anchors_size_for_training=2000):
        self.anchors_size_for_training = anchors_size_for_training
        self.anchors_size_for_training_in_half = anchors_size_for_training // 2

    def forward(self, anchors, gt_boxs):
        result_class = torch.zeros( (anchors.shape[0]) , device=device)
        result_bounding_delta = torch.zeros( (self.anchors_size_for_training, 4)).double().to(device)

        # 1. compute iou for every anchor & gt_box pair
        gt_boxs_len = gt_boxs.size()[0]
        anchors_len = anchors.size()[0]

        anchors_repeated = anchors.repeat(1, gt_boxs_len).view(-1, 4)
        gt_boxs_repeated = gt_boxs.repeat(anchors_len, 1)
        iou = get_iou(anchors_repeated, gt_boxs_repeated)
        iou = iou.view(anchors_len, gt_boxs_len)

        # 2. make iou > 0.7 anchor as positive and iou < 0.7 as negative
        _, anchor_iou_max_by_gt_index = torch.max(iou, dim=1)

        anchor_iou_max_by_gt = iou[ torch.arange(iou.size()[0]).to(device).long(), anchor_iou_max_by_gt_index ]

        positive_index = torch.nonzero(anchor_iou_max_by_gt[ anchor_iou_max_by_gt > 0.7 ]).squeeze(1)
        negative_index = torch.nonzero(anchor_iou_max_by_gt[ anchor_iou_max_by_gt < 0.3 ]).squeeze(1)
        # positive index and negative index balanced to 1:1
        positive_index_count = len(positive_index)
        negative_index_count = len(negative_index)

        diff = positive_index_count - negative_index_count

        def random_choice(tensor, size):
            perm = torch.randperm(tensor.size(0)).to(device)
            idx = perm[:size]
            return tensor[idx]

        if positive_index_count > self.anchors_size_for_training_in_half:
            positive_index = random_choice(positive_index, self.anchors_size_for_training_in_half)

        if negative_index_count > self.anchors_size_for_training_in_half:
            negative_index = random_choice(negative_index, self.anchors_size_for_training_in_half)

        if diff > 0:
            positive_index = random_choice(positive_index, negative_index_count)
        elif diff < 0:
            negative_index = random_choice(negative_index, positive_index_count)

        result_class[positive_index] = 1
        result_class[negative_index] = -1

        # just make sure every ground truth box got at least 1 positive anchor
        _, gt_iou_max_by_anchor_index = torch.max(iou, dim=0)
        result_class[gt_iou_max_by_anchor_index] = 1

        # 3. for every positive anchors, get the bounding refinement delta
        for i, index in enumerate(torch.nonzero(result_class == 1)):
            a = anchors[index].squeeze(0)
            gt = gt_boxs[ anchor_iou_max_by_gt_index[index] ].squeeze(0)

            # Convert coordinates to center plus width/height.
            # GT Box
            gt_h = gt[2] - gt[0]
            gt_w = gt[3] - gt[1]
            gt_center_y = gt[0] + 0.5 * gt_h
            gt_center_x = gt[1] + 0.5 * gt_w
            # Anchor
            a_h = a[2] - a[0]
            a_w = a[3] - a[1]
            a_center_y = a[0] + 0.5 * a_h
            a_center_x = a[1] + 0.5 * a_w

            # Compute the bbox refinement that the RPN should predict.
            result_bounding_delta[i] = torch.tensor(
                [
                    (gt_center_y - a_center_y) / a_h,
                    (gt_center_x - a_center_x) / a_w,
                    np.log(gt_h / a_h),
                    np.log(gt_w / a_w)
                ]
            , dtype=torch.float64, device=device )

            # Normalize
            # result_bounding_delta[i] /= config.RPN_BBOX_STD_DEV

        return result_class, result_bounding_delta

class MaskRCNN(nn.Module):

    def __init__(self, input_depth=3, num_classes=21, pool_size=7, image_size=512):
        super().__init__()
        resnet = ResNet50().to(device)

        # create sub networks
        self.fpn = FPN(*resnet.layers(), out_planes=256).to(device)
        self.rpn = RPN(256, 512).to(device)
        self.classifier = Classifier(21).to(device)
        self.mask = Mask(21).to(device)

        # create anchors
        self.anchor_scales = [16, 32, 64, 128]
        self.anchor_ratios = [0.5, 1, 2]
        self.anchor_feature_stride = [4, 8, 16, 32]
        anchors = generate_pyramid_anchors(self.anchor_scales, self.anchor_ratios, image_size, self.anchor_feature_stride)
        self.anchors = torch.tensor(anchors, dtype=torch.float64, device=device)

        # create necessary functions
        self.region_proposal = RegionProposal().forward
        self.rpn_target_generate = GenerateRPNTargets().forward
        self.target_generate = GenerateTarget().forward

    def forward(self, image, gt_bboxes, gt_labels, gt_masks):
        image = image.permute(2, 0, 1).unsqueeze(0)
        fpn_feature_maps = self.fpn(image)
        bbox_deltas, scores, class_logits = self.generate_regions(fpn_feature_maps)
        rois = self.region_proposal(bbox_deltas, scores, self.anchors).detach()
        target_rpn_classes, target_rpn_bbox_deltas = self.rpn_target_generate(self.anchors, gt_bboxes)
        rois, target_bboxes, target_masks, target_classes = self.target_generate(rois, gt_bboxes, gt_masks, gt_labels)
        pred_class_logits, pred_probs, pred_bboxes = self.classifier(fpn_feature_maps, rois)
        pred_masks = self.mask(fpn_feature_maps, rois)

        rpn_class_loss = compute_rpn_class_loss(target_rpn_classes, class_logits)
        rpn_bbox_loss = compute_rpn_bbox_loss(target_rpn_bbox_deltas, bbox_deltas, target_rpn_classes)
        mrcnn_class_loss = compute_mrcnn_class_loss(  target_classes, pred_class_logits )
        mrcnn_bbox_loss = compute_mrcnn_bbox_loss(target_bboxes, target_classes, pred_bboxes)
        mrcnn_mask_loss = compute_mrcnn_mask_loss(target_masks, target_classes, pred_masks)

        loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

        return loss

    def generate_regions(self, fpn_feature_maps):
        rpn_regions = []
        for o in fpn_feature_maps:
            rpn_regions.append( self.rpn(o) )
        bbox_deltas, class_logits, softmax = zip(*rpn_regions)

        def merge(x, y):
            return torch.cat( (x, y), dim=1 )
        bbox_deltas = functools.reduce(merge, bbox_deltas[1:], bbox_deltas[0])
        scores = functools.reduce(merge, class_logits[1:], class_logits[0])
        class_logits = functools.reduce(merge, class_logits[1:], class_logits[0])

        return bbox_deltas, scores, class_logits


def compute_rpn_class_loss(ground_truth, logits):
    """ground truth is (batch_size, anchors_size), anchors_size got posible 3 value:
    -1: represents negative one, 0: ignore, 1: positive one
    """
    ground_truth = ground_truth.squeeze(0)
    logits = logits.squeeze(0)

    valid_idx = torch.nonzero(ground_truth != 0).squeeze(1)

    ground_truth = ground_truth[valid_idx]
    logits = logits[valid_idx]

    ground_truth = (ground_truth == 1).long()
    return F.cross_entropy(logits, ground_truth)

def compute_rpn_bbox_loss(ground_truth, predicts, rpn_class):
    """pick one positive anchors and get the coorsponding predicts
    then pick the same number from ground truth and compute the loss
    """
    predicts = predicts.squeeze(0)
    positive_index = torch.nonzero(rpn_class == 1).squeeze(1)
    predicts = predicts[positive_index]
    ground_truth = ground_truth[:predicts.size()[0]]

    return F.smooth_l1_loss(predicts, ground_truth)

def compute_mrcnn_class_loss(ground_truth, mrcnn_logits):
    return F.cross_entropy(mrcnn_logits, ground_truth)

def compute_mrcnn_bbox_loss(gt_bbox, gt_class_id, mrcnn_bbox):
    """Only positive bbox and with the right class_id contribute to loss
    """
    positive_index = torch.nonzero(gt_class_id).squeeze(1)
    class_ids = gt_class_id[positive_index]
    targets = gt_bbox[positive_index, :]
    predicts = mrcnn_bbox[positive_index, class_ids, :]

    return F.smooth_l1_loss(predicts, targets)

def compute_mrcnn_mask_loss(gt_mask, gt_class_id, mrcnn_mask):
    """Only positive mask and with the right class_id contribute to loss
    """
    positive_index = torch.nonzero(gt_class_id).squeeze(1)
    class_ids = gt_class_id[positive_index]
    targets = gt_mask[positive_index, :]
    predicts = mrcnn_mask[positive_index, class_ids, :]
    return F.binary_cross_entropy(predicts, targets)
