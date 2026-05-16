from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO("yolo26l.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolo26l.engine'