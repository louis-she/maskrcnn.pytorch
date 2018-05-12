import functools

import numpy as np
import torch

from voc2012_dataset.dataset import VOC2012ClassSegmentation
import model
import utils

def train_epoch(dataset, model, optimizer, epoch_number):
    optimizer.zero_grad()
    for index, (image, gt_bboxes, gt_labels, gt_masks) in enumerate(voc_dataset):
        if index <= 1:
            continue
        image = torch.tensor(image, dtype=torch.float64, device=device)
        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float64, device=device)
        gt_labels = torch.tensor(gt_labels, dtype=torch.float64, device=device)
        gt_masks = torch.tensor(gt_masks, dtype=torch.float64, device=device)

        loss = model(image, gt_bboxes, gt_labels, gt_masks)
        print("epoch {} - batch {} - loss {}".format(epoch_number, index, loss))
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
    model = model.MaskRCNN()
    optimizer = create_optimizer(model)
    epoch = 10
    for i in range(epoch):
        train_epoch(voc_dataset, model, optimizer, i)
