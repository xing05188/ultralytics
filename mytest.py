from ultralytics import YOLO

model = YOLO(r"yolov8n-face.pt")
print(model.info())  # 查看模型结构