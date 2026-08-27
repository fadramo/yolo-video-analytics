import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

video_path = "videos/video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("No se pudo abrir el video")
else:
    print("video abierto correctamente")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "outputs/tracked_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)

print("Writer abierto:", writer.isOpened())
print("FPS:", fps)
print("Width:", width)
print("Height:", height)

frame_count = 0
line_x = 480
previous_x = {}
counted_ids = set ()

while True:

    ret,frame=cap.read()

    if not ret:
        break

    frame_count += 1

    results = model.track(
        frame, 
        persist=True, 
        classes=[2,7],
        tracker="bytetrack.yaml",
        verbose=False
        )
    
    result = results[0]
    boxes = result.boxes

    for box in boxes:

        
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        class_name = result.names[class_id]

        
        """print(
        "Frame:", frame_count,
        "Clase:", class_name,
        "Confianza:", round(confidence, 3),
        "Track ID:", box.id
        )"""


        if box.id is None:
            continue

        track_id = int(box.id.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        center_x = (x1 + x2) / 2
        previous_center_x = previous_x.get(track_id)

        if previous_center_x is not None:
            if ( 
                 previous_center_x > line_x 
                 and center_x <= line_x
                 and track_id not in counted_ids
                ): 

                counted_ids.add(track_id)

                crossing_number= len(counted_ids)

                print(
                "Cruce:",crossing_number, 
                "- Track ID:", track_id
                )

                evidence_path = (
                    f"outputs/evidence/"
                    f"crossing_{crossing_number:02d}_"
                    f"track_{track_id}_"
                    f"frame_{frame_count}.jpg"
                )

                cv2.imwrite(evidence_path, frame)


        previous_x[track_id] = center_x

        center_y = (y1 + y2) / 2

        
        """if class_name in ["car","truck"]:
            print(
                "Frame:", frame_count,
                "ID:", track_id,
                "Clase:", class_name,
                "Centro:", center_x, center_y,
                "Confianza", round(confidence,3)
            )"""


    annotated_frame = result.plot(
        line_width=1,
        font_size=10
    )

    writer.write(annotated_frame)

    cv2.line(
    annotated_frame,
    (line_x, 0),
    (line_x, height),
    (0, 255, 0),
    2
    )

    cv2.putText(
    annotated_frame,
    f"Carros: {len(counted_ids)}",
    (30, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
    )

    cv2.imshow("YOLO Tracking", annotated_frame)

    if cv2.waitKey(30)& 0xFF == ord("q"):
        break



   
    
print("Total de carros:", len(counted_ids))
print("Frames procesados:", frame_count)

cap.release()
writer.release()
cv2.destroyAllWindows()

 
    
