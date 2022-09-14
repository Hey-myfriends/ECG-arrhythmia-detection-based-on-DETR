# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import io
import torch
# from torchvision.ops.boxes import box_area


def box_cxw_to_xx(x):
    x_c, w = x.unbind(-1)
    b = [(x_c - 0.5 * w),
         (x_c + 0.5 * w)]
    return torch.stack(b, dim=-1)


def box_xx_to_cxw(x):
    x0, x1 = x.unbind(-1)
    b = [(x0 + x1) / 2,
         (x1 - x0)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    assert boxes.shape[-1] == 2, "box must be 2d..."
    assert (boxes[..., 1] >= boxes[..., 0]).all(), "box right point must greater than left..."
    return boxes[..., 1] - boxes[..., 0]

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1) ## [N,]
    area2 = box_area(boxes2) ## [M,]

    lt = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N,M,1]
    rb = torch.min(boxes1[:, None, 1], boxes2[:, 1])  # [N,M,1]

    wh = (rb - lt).clamp(min=0)  # [N,M,1]
    inter = wh.squeeze(-1)
    # inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter ## [N, M]

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 1] >= boxes1[:, 0]).all()
    assert (boxes2[:, 1] >= boxes2[:, 0]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, 0], boxes2[:, 0])
    rb = torch.max(boxes1[:, None, 1], boxes2[:, 1])

    wh = (rb - lt).clamp(min=0)  # [N,M]
    area = wh.squeeze(-1)
    # area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


if __name__ == "__main__":
    box1, box2 = torch.tensor([[1, 2], [4, 6], [10, 15]]), torch.tensor([[4, 5], [7, 8], [12, 13]])
    giou = generalized_box_iou(box1, box2)
    # print(iou.shape, area.shape)