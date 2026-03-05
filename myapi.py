import requests
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Deployment endpoint
url = "https://predict-6986e39d72c415cd0d3a-dproatj77a-de.a.run.app/predict"

# Headers with your deployment API key
headers = {"Authorization": "Bearer ul_23d23ec10278fe7a0559a0f02afbd9f85b125f66"}

# Inference parameters
data = {"conf": 0.25, "iou": 0.7, "imgsz": 640}

# Send image for inference
with open("C:/homework/ultralytics/mytest.jpg", "rb") as f:
    response = requests.post(url, headers=headers, data=data, files={"file": f})

result = response.json()
print(result)

# Load the original image
image_path = "C:/homework/ultralytics/mytest.jpg"
image = Image.open(image_path)
draw = ImageDraw.Draw(image)

# Check if predictions exist in the response
if "images" in result and len(result["images"]) > 0 and "results" in result["images"][0]:
    # Get image dimensions
    img_width, img_height = image.size
    
    # Draw each prediction
    for pred in result["images"][0]["results"]:
        box = pred["box"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        confidence = pred["confidence"]
        class_name = pred.get("name", "unknown")
        
        # Convert to absolute coordinates
        abs_x1, abs_y1 = int(x1), int(y1)
        abs_x2, abs_y2 = int(x2), int(y2)
        
        # Draw bounding box
        draw.rectangle([(abs_x1, abs_y1), (abs_x2, abs_y2)], outline="red", width=3)
        
        # Draw label with confidence
        label = f"{class_name}: {confidence:.2f}"
        draw.text((abs_x1, abs_y1 - 20), label, fill="red")

# Display the image with predictions
plt.figure(figsize=(12, 8))
plt.imshow(np.array(image))
plt.axis('off')
plt.title("YOLO Detection Results")
plt.show()