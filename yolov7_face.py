import os
import sys
import numpy
import torch
from model.det import Detection
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, xyxy2xywh, scale_coords


class YOLOv7Face(Detection):
    def __init__(self, params):
        super().__init__(params)
        self.model = attempt_load(params["model_path"], map_location=params["device"])
        self.device = params["device"]
        self.image_size = params["image_size"]
        self.conf_thresh = params["conf_thresh"]
        self.iou_thresh = params["iou_thresh"]

    @staticmethod
    def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
        coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
        coords[:, :10] /= gain
        # clip_coords(coords, img0_shape)
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
        coords[:, 4].clamp_(0, img0_shape[1])  # x3
        coords[:, 5].clamp_(0, img0_shape[0])  # y3
        coords[:, 6].clamp_(0, img0_shape[1])  # x4
        coords[:, 7].clamp_(0, img0_shape[0])  # y4
        coords[:, 8].clamp_(0, img0_shape[1])  # x5
        coords[:, 9].clamp_(0, img0_shape[0])  # y5
        return coords

    @staticmethod
    def dynamic_resize(shape, stride=64):
        max_size = max(shape[0], shape[1])
        if max_size % stride != 0:
            max_size = (int(max_size / stride) + 1) * stride
        return max_size

    def detect(self, image):
        imgsz = self.image_size
        if imgsz <= 0:  # original size
            imgsz = self.dynamic_resize(image.shape)
        imgsz = check_img_size(imgsz, s=64)  # check img_size
        img = letterbox(image, imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = numpy.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh)[0]
        gn = torch.tensor(image.shape)[[1, 0, 1, 0]].to(self.device)  # normalization gain whwh
        gn_lks = torch.tensor(image.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(self.device)  # normalization gain landmarks
        boxes = []
        h, w, c = image.shape
        if pred is not None:
            pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], image.shape).round()
            pred[:, 5:15] = self.scale_coords_landmarks(img.shape[2:], pred[:, 5:15], image.shape).round()
            for j in range(pred.size()[0]):
                xywh = (xyxy2xywh(pred[j, :4].view(1, 4)) / gn).view(-1)
                xywh = xywh.data.cpu().numpy()
                conf = float(pred[j, 4].cpu().numpy())
                class_num = int(pred[j, 15].cpu().numpy())
                x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
                y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
                x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
                y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
                boxes.append(["face", class_num, conf, x1, y1, x2 - x1, y2 - y1])

        return boxes