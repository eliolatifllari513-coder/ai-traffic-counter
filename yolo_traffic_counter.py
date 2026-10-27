import cv2
import numpy as np

# Ngarkojmë modelin YOLOv8n ONNX përmes OpenCV DNN
net = cv2.dnn.readNetFromONNX("yolov8n.onnx")

# COCO dataset: klasa 2 është "car", 5 është "bus", 7 është "truck"
VEHICLE_CLASSES = [2, 5, 7]

cap = cv2.VideoCapture("traffik.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 kërkon madhësi standarde 640x640
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), [0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()[0]

    # KËTU ISHTE GABIMI: Rrotullojmë matricën me .T që OpenCV ta lexojë saktë
    outputs = outputs.T

    boxes = []
    confidences = []
    h_orig, w_orig = frame.shape[:2]

    for i in range(outputs.shape[0]):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        class_id = np.argmax(classes_scores)

        # Kontrollojmë nëse është mjet transporti dhe ka siguri > 50%
        if max_score > 0.5 and class_id in VEHICLE_CLASSES:
            cx, cy, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            
            x = int((cx - w / 2) * w_orig / 640)
            y = int((cy - h / 2) * h_orig / 640)
            width = int(w * w_orig / 640)
            height = int(h * h_orig / 640)

            boxes.append([x, y, width, height])
            confidences.append(float(max_score))

    # Aplikojmë Non-Max Suppression për të eliminuar kutitë e dyfishta
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    vehicle_count = 0

    if len(indices) > 0:
        for i in indices.flatten():
            vehicle_count += 1
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Mjet: {confidences[i]*100:.1f}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Total Mjete (YOLOv8): {vehicle_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("YOLOv8 Traffic Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
