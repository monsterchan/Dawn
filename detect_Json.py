import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, 
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
# resultClsNum = {'person':0, 'bicycle':0, 'car':0, 'motorcycle':0, 'airplane':0, 'bus':0, 'train':0, 'truck':0, 'boat':0, 'traffic light':0,
#         'fire hydrant':0, 'stop sign':0, 'parking meter':0, 'bench':0, 'bird':0, 'cat':0, 'dog':0, 'horse':0, 'sheep':0, 'cow':0,
#         'elephant':0, 'bear':0, 'zebra':0, 'giraffe':0, 'backpack':0, 'umbrella':0, 'handbag':0, 'tie':0, 'suitcase':0, 'frisbee':0,
#         'skis':0, 'snowboard':0, 'sports ball':0, 'kite':0, 'baseball bat':0, 'baseball glove':0, 'skateboard':0, 'surfboard':0,
#         'tennis racket':0, 'bottle':0, 'wine glass':0, 'cup':0, 'fork':0, 'knife':0, 'spoon':0, 'bowl':0, 'banana':0, 'apple':0,
#         'sandwich':0, 'orange':0, 'broccoli':0, 'carrot':0, 'hot dog':0, 'pizza':0, 'donut':0, 'cake':0, 'chair':0, 'couch':0,
#         'potted plant':0, 'bed':0, 'dining table':0, 'toilet':0, 'tv':0, 'laptop':0, 'mouse':0, 'remote':0, 'keyboard':0, 'cell phone':0,
#         'microwave':0, 'oven':0, 'toaster':0, 'sink':0, 'refrigerator':0, 'book':0, 'clock':0, 'vase':0, 'scissors':0, 'teddy bear':0,
#         'hair drier':0, 'toothbrush':0}
class Detect:

    def __init__(self):
        self.weights = 'weights/yolov5s.pt'
        # self.weights = 'weights/yolov5x.pt'
        self.out = './inference/outImages'
        # Initialize

        self.device = select_device('')
        if os.path.exists(self.out):
            shutil.rmtree(self.out)  # delete outImages folder
        os.makedirs(self.out)  # make new outImages folder
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model

    def detect(self,images,save_img=False):
        resultCls = []
        resultJson = []
        source = images
        imgsz = 640

        # webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
        webcam = False


        # model = models  # load FP32 model
        imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Set Dataloader
        dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Run inference
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=True)[0]

            # Apply NMS

            pred = non_max_suppression(pred, 0.6, 0.5, agnostic=True)
            t2 = time_synchronized()



            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(self.out) / Path(p).name)
                txt_path = str(Path(self.out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                # s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # s += '%g %ss, ' % (n, names[int(c)])  # add to string
                        resultCls.append([int(c),names[int(c)],int(n)])


                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        # if save_txt:  # Write to file
                        resjson = []
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                            # with open(txt_path + '.txt', 'a') as f:
                            #     f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                        print("cls",cls)
                        # resjson.append(xyxy)
                        xywh.append(names[int(cls)])
                        xywh.append(float(conf))
                        resultJson.append(xywh)
                        # if save_img :  # Add bbox to image
                        #     label = '%s %.2f' % (names[int(cls)], conf)
                        #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        #     resultImage = im0
                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))
                # resultLable = s[8:]
                # Stream results
        return resultJson,resultCls

    #
    # if save_txt or save_img:
    #     print('Results saved to %s' % Path(out))
    #     if platform.system() == 'Darwin' and not opt.update:  # MacOS
    #         os.system('open ' + save_path)
    #
    # print('Done. (%.3fs)' % (time.time() - t0))


# if __name__ == '__main__':

    # print(opt)

    # with torch.no_grad():
        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
        #         detect()
        #         strip_optimizer(opt.weights)
        # else:
        #     detect()
        # detect()