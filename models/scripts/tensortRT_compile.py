"""
TensorRT export notes.

The two YOLO exports below are handled by Ultralytics.
Uncomment them only when you want to regenerate the YOLO TensorRT engines.

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

YuNet face detector TensorRT command.

Run this from the project root if you want to generate the optional engine file:

trtexec --onnx=models\\face_detection_yunet_2023mar.onnx --saveEngine=models\\face_detection_yunet_2023mar.engine --fp16

Notes:
- The app currently uses models/face_detection_yunet_2023mar.onnx through OpenCV FaceDetectorYN.
- OpenCV FaceDetectorYN cannot load a .engine file directly.
- The generated .engine is kept for a future dedicated TensorRT YuNet backend.
"""
