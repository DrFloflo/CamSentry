from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("yolo26m-pose.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolo26m-pose.engine'

"""
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace nvcr.io/nvidia/pytorch:24.12-py3 bash -lc "pip install --upgrade pip && pip install --force-reinstall 'numpy==1.26.4' 'opencv-python==4.10.0.84' && pip install --no-deps ultralytics onnx && python tensortRT_compile.py"
"""