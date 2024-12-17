import os
from ultralytics import YOLO, RTDETR

os.environ["WANDB_MODE"] = "offline"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#model = YOLO("yolov5m.pt")
#model = YOLO("yolov8l-worldv2.pt")
EPS = 47
BS = 16
IMZ = 640
MOSAIC = float(os.environ["Z_MOSAIC"])
device = [0]
MODEL = 'rtdetr-l.pt'
model = RTDETR(MODEL) #rtdetr-resnet101.yaml
model.train(
    data="data0.yaml", epochs=EPS, imgsz=IMZ, \
    device=device, batch=BS, plots=True, \
    flipud=0.3,mixup=0.2, \
    erasing=0.4,copy_paste=0.0, \
    #optimizer='AdamW', \
    hsv_s=0.0, hsv_v=0.0, \
    mosaic=MOSAIC, \
    close_mosaic=20, \
    #box=0.5, cls=0.05,\
)
