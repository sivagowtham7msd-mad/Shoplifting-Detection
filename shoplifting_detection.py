from ultralytics import YOLO
import mediapipe as mp
import numpy as np
import imutils
import cv2
import os
from collections import deque

from config.parameters import (
    WIDTH, shoplifting_status, not_shoplifting_status, no_detection_status,
    cls0_rect_color, cls1_rect_color, conf_color, status_color_ok, status_color_bad,
    frame_name, quit_key,
    ALERT_SCORE_THRESHOLD,
    SCORE_REACH_TO_CONCEAL, SCORE_REACH_TO_BAG,
    SCORE_REPEATED_REACH, SCORE_ABNORMAL_MOTION,
    HAND_HISTORY_FRAMES, ABNORMAL_MOTION_PX, REPEATED_REACH_COUNT,
    REACH_ZONE_TOP, REACH_ZONE_BOT,
    CONCEAL_ZONE_TOP, CONCEAL_ZONE_BOT,
    BAG_ZONE_TOP, BAG_ZONE_BOT,
)

# ── I/O ───────────────────────────────────────────────────────────────────────
output_path = "Samples/outputs/sr1_output.avi"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────
yolo_model = YOLO("configs/shoplifting_weights.pt")
print(f"[INFO] YOLO classes: {yolo_model.names}")

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_model = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ── Capture ───────────────────────────────────────────────────────────────────
# Swap to local file if needed:
# cap = cv2.VideoCapture("Samples/inputs/sr1.mp4")
rtsp_url = "rtsp://admin:MVGBCE@10.135.113.177:554/ch1/main"
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print(f"[ERROR] Could not open stream: {rtsp_url}")
    exit()

writer = None


# ── Per-person state (keyed by detection index within a frame) ────────────────
# Each entry: deque of (wrist_x, wrist_y, zone_label)
hand_histories   = {}   # idx -> deque(maxlen=HAND_HISTORY_FRAMES)
reach_counts     = {}   # idx -> int  (how many reach events seen)
suspicion_scores = {}   # idx -> float


def get_zone(norm_y, box_top, box_bot):
    """Map a normalised wrist y-coordinate to a body-relative zone label."""
    box_h = box_bot - box_top
    if box_h == 0:
        return "unknown"
    rel = (norm_y - box_top) / box_h          # 0 = top of person, 1 = bottom
    if rel < REACH_ZONE_TOP:
        return "head"
    elif rel < REACH_ZONE_BOT:
        return "reach"
    elif rel < CONCEAL_ZONE_BOT:
        return "conceal"
    else:
        return "bag"


def compute_suspicion(history, reach_count):
    """
    Analyse hand history and return a suspicion score (0-100).
    history: deque of (x, y, zone) tuples
    """
    score = 0
    zones = [z for _, _, z in history]

    # 1. Reach → Conceal transition
    for i in range(1, len(zones)):
        if zones[i - 1] == "reach" and zones[i] == "conceal":
            score += SCORE_REACH_TO_CONCEAL
            break   # count once per window

    # 2. Reach → Bag transition
    for i in range(1, len(zones)):
        if zones[i - 1] == "reach" and zones[i] == "bag":
            score += SCORE_REACH_TO_BAG
            break

    # 3. Repeated reaching
    if reach_count >= REPEATED_REACH_COUNT:
        score += SCORE_REPEATED_REACH

    # 4. Abnormal / erratic wrist motion
    coords = [(x, y) for x, y, _ in history]
    if len(coords) >= 2:
        displacements = [
            np.hypot(coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1])
            for i in range(1, len(coords))
        ]
        if np.mean(displacements) > ABNORMAL_MOTION_PX:
            score += SCORE_ABNORMAL_MOTION

    return min(score, 100)


# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=WIDTH)
    h_frame, w_frame = frame.shape[:2]

    # Init writer on first frame
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, 25, (w_frame, h_frame), True)

    overall_status       = no_detection_status
    any_shoplifting      = False

    # ── Step 1: Person detection with YOLO ───────────────────────────────────
    results  = yolo_model.predict(frame, conf=0.4, verbose=False)
    boxes    = results[0].boxes
    cc_data  = np.array(boxes.data) if len(boxes) else np.array([])

    current_indices = set()

    if len(cc_data):
        xyxy_all = np.array(boxes.xyxy).astype("int32")

        for idx, ((x1, y1, x2, y2), det) in enumerate(zip(xyxy_all, cc_data)):
            conf_det = float(det[4])
            current_indices.add(idx)

            # Init state for new person
            if idx not in hand_histories:
                hand_histories[idx]   = deque(maxlen=HAND_HISTORY_FRAMES)
                reach_counts[idx]     = 0
                suspicion_scores[idx] = 0

            # ── Step 2: Pose estimation on the person crop ────────────────────
            pad = 10
            px1 = max(0, x1 - pad)
            py1 = max(0, y1 - pad)
            px2 = min(w_frame, x2 + pad)
            py2 = min(h_frame, y2 + pad)
            person_crop = frame[py1:py2, px1:px2]

            wrist_zone = "unknown"
            wrist_px, wrist_py = None, None

            if person_crop.size > 0:
                rgb_crop   = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                pose_result = pose_model.process(rgb_crop)

                if pose_result.pose_landmarks:
                    lm = pose_result.pose_landmarks.landmark
                    crop_h, crop_w = person_crop.shape[:2]

                    # Use the wrist with higher visibility (more likely to be active hand)
                    lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                    rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
                    wrist = lw if lw.visibility >= rw.visibility else rw

                    # Convert to frame-absolute pixel coords
                    wrist_px = int(px1 + wrist.x * crop_w)
                    wrist_py = int(py1 + wrist.y * crop_h)

                    # Get body-relative zone
                    wrist_zone = get_zone(wrist_py, y1, y2)

                    # Track reach events
                    if wrist_zone == "reach":
                        reach_counts[idx] += 1

                    hand_histories[idx].append((wrist_px, wrist_py, wrist_zone))

                    # Draw pose skeleton on frame
                    mp_drawing.draw_landmarks(
                        frame[py1:py2, px1:px2],
                        pose_result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1)
                    )

            # ── Step 3: Compute suspicion score ───────────────────────────────
            score = compute_suspicion(hand_histories[idx], reach_counts[idx])
            suspicion_scores[idx] = score
            is_suspicious = score >= ALERT_SCORE_THRESHOLD

            # ── Step 4: Draw bounding box and overlays ────────────────────────
            box_color = cls1_rect_color if is_suspicious else cls0_rect_color
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Confidence + score label
            label = f"{conf_det*100:.1f}%  score:{score}"
            cv2.putText(frame, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, conf_color, 1)

            # Zone label near wrist
            if wrist_px is not None:
                cv2.circle(frame, (wrist_px, wrist_py), 5, (0, 0, 255), -1)
                cv2.putText(frame, wrist_zone, (wrist_px + 6, wrist_py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw zone guide lines on person box
            for ratio, color in [
                (REACH_ZONE_TOP,   (200, 200, 0)),
                (REACH_ZONE_BOT,   (0, 200, 200)),
                (CONCEAL_ZONE_BOT, (0, 100, 255)),
            ]:
                line_y = int(y1 + ratio * (y2 - y1))
                cv2.line(frame, (x1, line_y), (x2, line_y), color, 1)

            if is_suspicious:
                any_shoplifting = True
                overall_status  = shoplifting_status
            elif overall_status == no_detection_status:
                overall_status = not_shoplifting_status

    # Prune state for persons no longer in frame
    for old_idx in list(hand_histories.keys()):
        if old_idx not in current_indices:
            del hand_histories[old_idx]
            del reach_counts[old_idx]
            del suspicion_scores[old_idx]

    # ── Step 5: Draw global status bar ───────────────────────────────────────
    status_color = status_color_bad if any_shoplifting else status_color_ok
    cv2.rectangle(frame, (0, 0), (w_frame, 28), (0, 0, 0), -1)
    cv2.putText(frame, overall_status, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    cv2.imshow(frame_name, frame)
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord(quit_key):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
pose_model.close()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
print(f"[INFO] Output saved to {output_path}")
