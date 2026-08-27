from ultralytics import YOLO
model= YOLO("yolo26n.pt")

results= model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True
)


result = results[0]
boxes = result.boxes
detections = []

for box in boxes:
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())
    class_name = result.names[class_id]

    x1, y1, x2, y2 = box.xyxy[0].tolist()
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    if confidence >= 0.60 and class_name == "person":
        detection = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "center_x": center_x,
            "center_y": center_y,
            "bbox": [x1,y1,x2,y2]
          }

        detections.append(detection)
        #print(class_name,confidence,center_x,center_y)

print(detections)
count = len(detections)
print("Personas detectadas:", count)

print("Deteccion terminada")