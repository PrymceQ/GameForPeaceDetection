# author：chouti

import sys
from pathlib import Path
import cv2
import torch
import numpy as np
from PIL import ImageGrab

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.general import is_ascii, non_max_suppression, scale_coords, set_logging
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


@torch.no_grad()
def run(weights='./runs/train/hepingjingying96/weights/best.pt',  # model.pt path(s)  ########################################  要改成你自己新的训练好的权重模型文件，地址自己换一下
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        line_thickness=3,  # bounding box thickness (pixels)
        half=False,  # use FP16 half-precision inference
        ):

    # Initialize
    set_logging()
    device = select_device(device)
    print(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, map_location=device)  # load FP32 model
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)
    # capture = cv2.VideoCapture(0)



    # 我自己的电脑分辨率 1920*1080（参考）
    BOX = (0,0,1024,576)   ##############################################################################  定义扫描的电脑屏幕范围，这里设置的是整个电脑屏幕左上角，不喜欢自己改
    #左上角坐标和右下角坐标  #BOX=(0,0,1024,576) 1024-0宽 576-0高               要满足高宽都是32的倍数，否则会报错，96*n 和128*n 
    #调整box的值即可改变截取区域
    
    
    
    
    while True:
        # 获取一帧
        frame = np.array(ImageGrab.grab(bbox=BOX)) 
        
        img = torch.from_numpy(frame).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        img = img.transpose(2, 3)
        img = img.transpose(1, 2)

        # Inference
        pred = model(img, augment=False, visualize=False)[0]

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # detections per image
            s = ''
            annotator = Annotator(frame, line_width=line_thickness, pil=not ascii)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += str(n.item()) + ' ' + str(names[int(c)]) + ' '  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    print(xyxy)

            print('result:'+s)

        cv2.imshow('window', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(25) & 0xFF ==ord("q"):
            cv2.destroyAllWindows()
            break


def main():
    run()


if __name__ == "__main__":
    main()

