"""
general utils
"""

from pathlib import Path
import numpy as np
import torch
import requests
from PIL import Image
from torch.cuda import amp
from yolov5.utils.datasets import exif_transpose, letterbox
from yolov5.utils.general import (make_divisible, non_max_suppression,
                                  scale_coords, cv2)
from yolov5.utils.torch_utils import time_sync
from yolov5.models.common import Detections
from yolov5.models.common import AutoShape



#problems with imread
#https://github.com/ultralytics/yolov5/pull/7287
def imread(path, flags=cv2.IMREAD_COLOR):
    """
    the function imread of cv to be redfined
    #problems with imread
    #https://github.com/ultralytics/yolov5/pull/7287
    """
    return cv2.imdecode(np.fromfile(path, np.uint8), flags)


cv2.imread = imread # redefine


@torch.no_grad()
def forward(self, imgs, size=640, augment=False, profile=False):
    """
    Inference from various sources. For height=640, width=1280, RGB images example inputs are:
      file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
      URI:             = 'https://ultralytics.com/images/zidane.jpg'
      OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
      PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
      numpy:           = np.zeros((640,1280,3))  # HWC
      torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
      multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images
    """

    t = [time_sync()]
    p = next(self.model.parameters()) if self.pt else torch.zeros(1)  # for device and type
    autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
    if isinstance(imgs, torch.Tensor):  # torch
        with amp.autocast(autocast):
            return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

    # Pre-process
    n, imgs = (len(imgs), list(imgs)) if isinstance(imgs, (list, tuple)) else (1, [imgs])  # number, list of images
    shape0, shape1, files = [], [], []  # image and inference shapes, filenames
    for i, im in enumerate(imgs):
        f = f'image{i}'  # filename
        if isinstance(im, (str, Path)):  # filename or uri
            im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
            im = np.asarray(exif_transpose(im))
            im = 20 * np.log10(im)
        elif isinstance(im, Image.Image):  # PIL Image
            im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
        files.append(Path(f).with_suffix('.jpg').name)
        if im.shape[0] < 5:  # image in CHW
            im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
        im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
        s = im.shape[:2]  # HWC
        shape0.append(s)  # image shape
        g = (size / max(s))  # gain
        shape1.append([y * g for y in s])
        imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
    shape1 = [make_divisible(x, self.stride) if self.pt \
        else size for x in np.array(shape1).max(0)]  # inf shape
    x = [letterbox(im, shape1, auto=False)[0] for im in imgs]  # pad
    x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
    x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
    t.append(time_sync())

    with amp.autocast(autocast):
        # Inference
        y = self.model(x, augment, profile)  # forward
        t.append(time_sync())

        # Post-process
        y = non_max_suppression(y if self.dmb else y[0],
                                self.conf,
                                self.iou,
                                self.classes,
                                self.agnostic,
                                self.multi_label,
                                max_det=self.max_det)  # NMS
        for i in range(n):
            scale_coords(shape1, y[i][:, :4], shape0[i])

        t.append(time_sync())
        return Detections(imgs, y, files, t, self.names, x.shape)

# Overriding mehtod for image reading
AutoShape.forward = forward
