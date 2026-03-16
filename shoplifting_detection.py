

from ultralytics import YOLO
import torch
import numpy as np
import imutils
import cv2
import os


from config.parameters import WIDTH, start_status, shoplifting_status, not_shoplifting_status
from config.parameters import cls0_rect_color, cls1_rect_color, conf_color, status_color
from config.parameters import quit_key, frame_name


def load_trusted_yolo_model(weights_path):
    original_torch_load = torch.load

    def compatible_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = compatible_torch_load
    try:
        return YOLO(weights_path)
    finally:
        torch.load = original_torch_load

input_path = "Samples/inputs/sr1.mp4"
output_path = "Samples/outputs/sr1_output.avi"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
mymodel = load_trusted_yolo_model("configs/shoplifting_weights.pt")

#For videos
# cap = cv2.VideoCapture(input_path)

# For Laptop camera
rtsp_url="rtsp://admin:MVGBCE@10.135.113.177:554/ch1/main"
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print(f"[ERROR] Could not open RTSP stream: {rtsp_url}")
    exit()

writer = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=WIDTH)
    status = start_status

    result = mymodel.predict(frame, conf=0.4)
    cc_data = np.array(result[0].boxes.data)

    if len(cc_data) != 0:
        xywh = np.array(result[0].boxes.xywh).astype("int32")
        xyxy = np.array(result[0].boxes.xyxy).astype("int32")

        for (x1, y1, _, _), (_, _, w, h), (_, _, _, _, conf, clas) in zip(xyxy, xywh, cc_data):
            person = frame[y1:y1+h, x1:x1+w]

            if clas == 1:  
                cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), cls1_rect_color, 2)
                x_center = int(x1 + w/2)
                cv2.circle(frame, (x_center, y1), 6, (0, 0, 255), 8)
                text = f"{np.round(conf*100,2)}%"
                cv2.putText(frame, text, (x1+10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)
                status = shoplifting_status

            elif clas == 0 and conf > 0.5:  
                cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), cls0_rect_color, 1)
                text = f"{np.round(conf*100,2)}%"
                cv2.putText(frame, text, (x1+10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)
                status = not_shoplifting_status

   
    cv2.putText(frame, status, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.imshow(frame_name, frame)

    
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord(quit_key):
        break

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
print(f"[INFO] Video saved to {output_path}") 

