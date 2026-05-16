from ultralytics import YOLO
YOLO('yolo26m-pose.pt', task='pose').export(format='engine', device=0)