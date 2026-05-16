from ultralytics import YOLO

YOLO('models/yolo26m.pt', task='detect').export(
    format='engine',
    device=0,
    half=True,
    simplify=True)

YOLO('models/yolo26m-seg.pt', task='segment').export(
    format='engine',
    device=0,
    half=True,
    simplify=True)