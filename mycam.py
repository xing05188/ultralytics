from ultralytics import YOLO

import cv2

model = YOLO(r'C:\homework\ultralytics\runs\detect\train3\weights\best.pt')
results = model(
    source=0,
    stream=True,
)

for result in results:
    plot = result.plot()
    cv2.imshow('result', plot)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
