from inference.models.utils import get_model
import cv2

from ultralytics import YOLO
import cv2



model = get_model(
    model_id="playing-cards-ow27d/4",
    api_key="MKhkXW64EgVY7kD2RnTN"
)



import cv2
import supervision as sv


box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0) 

card_count = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imwrite("temp.jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    result = model.infer(frame_rgb)[0]
    detections = sv.Detections.from_inference(result.dict(by_alias=True, exclude_none=True))
    labels = [p.class_name for p in result.predictions]

    card_count.add(str(detections))

    annotated = box_annotator.annotate(scene=frame, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    cv2.imshow(".", annotated)

    count = len(card_count)
    cv2.putText(
        annotated,
        f"Hand count: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        10,  
        (0, 255, 0),  
        12,  
        cv2.LINE_AA
    )

    

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


