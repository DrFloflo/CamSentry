from ultralytics import YOLO

YOLO('models/yolo26m-pose.pt', task='pose').export(
    format='engine',
    device=0,
    half=True,
    simplify=True)

YOLO('models/yolo26l.pt', task='detect').export(
    format='engine',
    device=0,
    half=True,
    simplify=True)