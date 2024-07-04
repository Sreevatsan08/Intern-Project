import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')
model.eval()

classes = ['Jeff Bezos', 'Elon Musk', 'Warren Buffett', 'Bill Gates', 'Mark Zuckerberg', 'Larry Page', 'Sergey Brin', 'Tim Cook', 'Satya Nadella', 'Jack Ma']

def detect(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    bboxes = results.xyxy[0].numpy()
    
    for bbox in bboxes:
        x1, y1, x2, y2, conf, cls = bbox
        label = f"{classes[int(cls)]} {conf:.2f}"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imshow('Object Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = 'path/to/your/image.jpg'
detect(image_path)




