from ultralytics import YOLO

model = YOLO(r"C:\homework\ultralytics\models\yolov6m-face.pt", task="detect")
out_file = model.export(
    format="engine",
    int8=True,
    dynamic=True,
    half=True,
    )
print("已导出模型到:", out_file)
# out_file = model.export(
#     format="engine",
#     imgsz=640,         # (int | list) input images size for exported model
#     batch=1,           # (int) batch size for exported model
#     keras=False,       # (bool) use Keras
#     optimize=False,    # (bool) TorchScript: optimize for mobile
#     half=False,        # (bool) ONNX/TF/TensorRT: FP16 quantization
#     int8=False,        # (bool) CoreML/TF/TensorRT/OpenVino INT8 quantization
#     dynamic=False,     # (bool) ONNX/TF/TensorRT: dynamic axes
#     simplify=False,    # (bool) ONNX: simplify model using `onnxslim`
#     opset=None,        # (int, optional) ONNX: opset version
#     workspace=4,       # (int) TensorRT: workspace size (GiB)
#     nms=False,         # (bool) CoreML: add NMS
# )
# reference https://docs.ultralytics.com/modes/export
