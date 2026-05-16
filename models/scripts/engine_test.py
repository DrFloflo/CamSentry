import tensorrt as trt

engine_path = "models/yolo26l.engine"
logger = trt.Logger(trt.Logger.WARNING)

with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

print("Engine:", engine_path)
print("num_io_tensors:", engine.num_io_tensors)

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    dtype = engine.get_tensor_dtype(name)
    shape = engine.get_tensor_shape(name)
    print(i, name, mode, dtype, shape)
