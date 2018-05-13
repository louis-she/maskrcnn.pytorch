import functools
import cProfile, pstats
import sys
import gc

import numpy as np
import torch

from voc2012_dataset.dataset import VOC2012ClassSegmentation
import model
from model import RoiNotMatched, AnchorNotMatched
import utils

extra_tensor = []

def check_tensor():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())

def train_epoch(dataset, model, optimizer, epoch_number):
    optimizer.zero_grad()
    for index, (image, gt_bboxes, gt_labels, gt_masks) in enumerate(voc_dataset):
        image = torch.tensor(image, dtype=torch.float64, device=device)
        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float64, device=device)
        gt_labels = torch.tensor(gt_labels, dtype=torch.float64, device=device)
        gt_masks = torch.tensor(gt_masks, dtype=torch.float64, device=device)

        try:
            loss, (rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss) = \
                    model(image, gt_bboxes, gt_labels, gt_masks)
        except AnchorNotMatched:
            print('[skipped!!] epoch {} - batch {} - reason {}'.format(epoch_number, index, 'no positive anchor'))
            continue
        except RoiNotMatched:
            print('[skipped!!] epoch {} - batch {} - reason {}'.format(epoch_number, index, 'no positive roi'))
            continue

        print("epoch {} - batch {} - loss {}, "
              "rpn_class_loss {}, rpn_bbox_loss {}, "
              "mrcnn_class_loss {}, mrcnn_bbox_loss {}, "
              "mrcnn_mask_loss {}".format(epoch_number, index, loss, rpn_class_loss, rpn_bbox_loss,
               mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def create_optimizer(model):
    trainables_wo_bn = [param for name, param in model.named_parameters() if param.requires_grad and not 'bn' in name]
    trainables_only_bn = [param for name, param in model.named_parameters() if param.requires_grad and 'bn' in name]

    return torch.optim.SGD([
        {'params': trainables_wo_bn, 'weight_decay': 0.0001},
        {'params': trainables_only_bn}
    ], lr=0.0001, momentum=0.9)

if __name__ == "__main__":
    # global configuration
    torch.set_default_tensor_type('torch.DoubleTensor')
    device = torch.device("cuda")

    voc_dataset = VOC2012ClassSegmentation('/home/louis/datasets/VOCdevkit/VOC2012')
    data_loader = torch.utils.data.DataLoader(voc_dataset, batch_size=1, shuffle=False, num_workers=4)
    model = model.MaskRCNN()
    model.load_state_dict(torch.load( './resnet50-19c8e357.pth' ), strict=False)

    for parameter in model.fpn.parameters():
        parameter.requires_grade = False

    optimizer = create_optimizer(model)
    epoch = 10
    for i in range(epoch):
        train_epoch(data_loader, model, optimizer, i)