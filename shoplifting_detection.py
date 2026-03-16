from alert import send_whatsapp_alert, start_ngrok
from ultralytics import YOLO
import numpy as np
import imutils
import cv2
import os
import datetime
import threading
import winsound
from collections import deque

from config.parameters import (
    WIDTH, shoplifting_status, not_shoplifting_status, no_detection_status,
    cls0_rect_color, cls1_rect_color, conf_color, status_color_ok, status_color_bad,
    frame_name, quit_key,
    ALERT_SCORE_THRESHOLD,
    SCORE_REACH_TO_CONCEAL, SCORE_REACH_TO_BAG, SCORE_REPEATED_REACH,
    SCORE_ABNORMAL_MOTION, SCORE_BOTH_HANDS, SCORE_LOITERING,
    HAND_HISTORY_FRAMES, ABNORMAL_MOTION_RATIO, REPEATED_REACH_COUNT,
    REACH_DWELL_FRAMES, TRACK_GRACE_FRAMES,
    ZONE_SMOOTH_FRAMES, LOITER_FRAMES, LOITER_MOVE_RATIO,
    REACH_ZONE_TOP, REACH_ZONE_BOT, CONCEAL_ZONE_BOT,
    SNAPSHOT_DIR,
)

# ── COCO keypoint indices ─────────────────────────────────────────────────────
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW     = 7
KP_RIGHT_ELBOW    = 8
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10
VISIBILITY_THRESH = 0.45


def play_alarm():
    def _beep():
        for _ in range(5):
            winsound.Beep(1000, 400)
            winsound.Beep(800,  300)
    threading.Thread(target=_beep, daemon=True).start()


os.makedirs(SNAPSHOT_DIR, exist_ok=True)
start_ngrok(SNAPSHOT_DIR)
output_path = "Samples/outputs/sr1_output.avi"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

pose_model = YOLO("yolov8n-pose.pt")  # nano = best for CPU

# Swap to RTSP for live camera:
# cap = cv2.VideoCapture("rtsp://admin:PASSWORD@IP:554/ch1/main")
cap = cv2.VideoCapture("Samples/inputs/sr1.mp4")
if not cap.isOpened():
    print("[ERROR] Could not open video source")
    exit()

writer        = None
person_states = {}
next_tid      = 0


def iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0


def match_detections(states, new_boxes, iou_thresh=0.35):
    matched, used_tids = {}, set()
    unmatched = list(range(len(new_boxes)))
    for box_idx, box in enumerate(new_boxes):
        best_tid, best_iou = None, iou_thresh
        for tid, state in states.items():
            if tid in used_tids:
                continue
            s = iou(state['box'], box)
            if s > best_iou:
                best_iou, best_tid = s, tid
        if best_tid is not None:
            matched[best_tid] = box_idx
            used_tids.add(best_tid)
            unmatched.remove(box_idx)
    return matched, unmatched


def get_zone(wrist_y, box_top, box_bot):
    box_h = box_bot - box_top
    if box_h == 0:
        return "unknown"
    rel = (wrist_y - box_top) / box_h
    if   rel < REACH_ZONE_TOP:   return "head"
    elif rel < REACH_ZONE_BOT:   return "reach"
    elif rel < CONCEAL_ZONE_BOT: return "conceal"
    else:                        return "bag"


def smooth_zone(history, n=ZONE_SMOOTH_FRAMES):
    """Majority vote over last n zone readings — reduces jitter false positives."""
    if not history:
        return "unknown"
    recent = [z for _, _, z in list(history)[-n:]]
    return max(set(recent), key=recent.count)


def is_loitering(centroid_hist, box_h):
    """True if person barely moved over LOITER_FRAMES — suspicious near shelf."""
    if len(centroid_hist) < LOITER_FRAMES:
        return False
    pts = list(centroid_hist)[-LOITER_FRAMES:]
    spread = np.hypot(max(p[0] for p in pts) - min(p[0] for p in pts),
                      max(p[1] for p in pts) - min(p[1] for p in pts))
    return spread < LOITER_MOVE_RATIO * box_h


def compute_suspicion(left_hist, right_hist, reach_entries, box_h, centroid_hist):
    abnormal_px = ABNORMAL_MOTION_RATIO * box_h

    def score_hand(history):
        s = 0
        zones = [z for _, _, z in history]
        for i in range(1, len(zones)):
            if zones[i] == "conceal" and "reach" in zones[max(0, i-15):i]:
                s += SCORE_REACH_TO_CONCEAL
                break
        for i in range(1, len(zones)):
            if zones[i] == "bag" and "reach" in zones[max(0, i-15):i]:
                s += SCORE_REACH_TO_BAG
                break
        coords = [(x, y) for x, y, _ in history]
        if len(coords) >= 2:
            disps = [np.hypot(coords[i][0]-coords[i-1][0],
                              coords[i][1]-coords[i-1][1])
                     for i in range(1, len(coords))]
            if np.mean(disps) > abnormal_px:
                s += SCORE_ABNORMAL_MOTION
        return s

    # Sum both hands (catches two-handed concealment), cap at 80
    score = min(score_hand(left_hist) + score_hand(right_hist), 80)

    if reach_entries >= REPEATED_REACH_COUNT:
        score += SCORE_REPEATED_REACH

    # Both hands simultaneously in suspicious zones
    if smooth_zone(left_hist) in ("conceal", "bag") and \
       smooth_zone(right_hist) in ("conceal", "bag"):
        score += SCORE_BOTH_HANDS

    if is_loitering(centroid_hist, box_h):
        score += SCORE_LOITERING

    return min(int(score), 100)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=WIDTH)
    h_frame, w_frame = frame.shape[:2]

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, 25, (w_frame, h_frame), True)

    overall_status  = no_detection_status
    any_shoplifting = False
    clean_frame     = frame.copy()

    results   = pose_model.predict(frame, conf=0.35, verbose=False)
    result    = results[0]
    new_boxes = result.boxes.xyxy.cpu().numpy().astype("int32") if len(result.boxes) else np.zeros((0, 4), dtype="int32")
    kps       = result.keypoints.data.cpu().numpy() if result.keypoints is not None and len(result.keypoints.data) else None

    matched, unmatched_new = match_detections(person_states, new_boxes)

    for tid in list(person_states.keys()):
        if tid not in matched:
            person_states[tid]['missed_frames'] += 1
            if person_states[tid]['missed_frames'] > TRACK_GRACE_FRAMES:
                del person_states[tid]

    for box_idx in unmatched_new:
        person_states[next_tid] = {
            'box':           tuple(new_boxes[box_idx]),
            'left':          deque(maxlen=HAND_HISTORY_FRAMES),
            'right':         deque(maxlen=HAND_HISTORY_FRAMES),
            'centroid_hist': deque(maxlen=LOITER_FRAMES + 10),
            'reach_entries': 0,
            'frame_count':   0,
            'missed_frames': 0,
            'snapshot_taken': False,
            'dwell_left':    0,
            'dwell_right':   0,
        }
        matched[next_tid] = box_idx
        next_tid += 1

    for tid, box_idx in matched.items():
        x1, y1, x2, y2 = new_boxes[box_idx]
        state = person_states[tid]
        state['box']           = (x1, y1, x2, y2)
        state['missed_frames'] = 0
        state['frame_count']  += 1
        box_h    = max(y2 - y1, 1)
        conf_det = float(result.boxes.conf[box_idx].cpu())

        # Track centroid for loitering
        state['centroid_hist'].append(((x1+x2)//2, (y1+y2)//2))

        if kps is not None and box_idx < len(kps):
            person_kps = kps[box_idx]

            for kp_idx, hist_key, dwell_key in [
                (KP_LEFT_WRIST,  'left',  'dwell_left'),
                (KP_RIGHT_WRIST, 'right', 'dwell_right'),
            ]:
                kp = person_kps[kp_idx]
                wx, wy, conf_kp = float(kp[0]), float(kp[1]), float(kp[2])
                if conf_kp < VISIBILITY_THRESH:
                    state[dwell_key] = 0
                    continue
                wx, wy = int(wx), int(wy)
                zone = get_zone(wy, y1, y2)
                state[hist_key].append((wx, wy, zone))
                if zone == "reach":
                    state[dwell_key] += 1
                    if state[dwell_key] == REACH_DWELL_FRAMES:
                        state['reach_entries'] += 1
                else:
                    state[dwell_key] = 0
                dot_color = (0, 255, 0) if hist_key == 'left' else (255, 100, 0)
                cv2.circle(frame, (wx, wy), 6, dot_color, -1)
                cv2.putText(frame, smooth_zone(state[hist_key]),
                            (wx + 7, wy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            for w_idx, e_idx, color in [
                (KP_LEFT_WRIST,  KP_LEFT_ELBOW,  (0, 255, 0)),
                (KP_RIGHT_WRIST, KP_RIGHT_ELBOW, (255, 100, 0)),
            ]:
                wkp = person_kps[w_idx]; ekp = person_kps[e_idx]
                if wkp[2] >= VISIBILITY_THRESH and ekp[2] >= VISIBILITY_THRESH:
                    cv2.line(frame, (int(wkp[0]), int(wkp[1])),
                             (int(ekp[0]), int(ekp[1])), color, 2)

            for s_idx, e_idx, color in [
                (KP_LEFT_SHOULDER,  KP_LEFT_ELBOW,  (0, 200, 100)),
                (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, (200, 100, 0)),
            ]:
                skp = person_kps[s_idx]; ekp = person_kps[e_idx]
                if skp[2] >= VISIBILITY_THRESH and ekp[2] >= VISIBILITY_THRESH:
                    cv2.line(frame, (int(skp[0]), int(skp[1])),
                             (int(ekp[0]), int(ekp[1])), color, 1)

        score = compute_suspicion(state['left'], state['right'],
                                  state['reach_entries'], box_h,
                                  state['centroid_hist'])
        is_suspicious = score >= ALERT_SCORE_THRESHOLD

        if state['frame_count'] % 15 == 0:
            print(f"[INFO] tid={tid}  score={score}  reaches={state['reach_entries']}")

        box_color = cls1_rect_color if is_suspicious else cls0_rect_color
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, f"{conf_det*100:.1f}%  score:{score}",
                    (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, conf_color, 1)

        for ratio, lc in [(REACH_ZONE_TOP, (200, 200, 0)),
                          (REACH_ZONE_BOT,  (0, 200, 200)),
                          (CONCEAL_ZONE_BOT,(0, 100, 255))]:
            ly = int(y1 + ratio * (y2 - y1))
            cv2.line(frame, (x1, ly), (x2, ly), lc, 1)

        if is_suspicious:
            if not state['snapshot_taken']:
                state['snapshot_taken'] = True
                ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cx1  = max(0, x1-10);       cy1 = max(0, y1-10)
                cx2  = min(w_frame, x2+10); cy2 = min(h_frame, y2+10)
                snap = clean_frame[cy1:cy2, cx1:cx2].copy()
                cv2.rectangle(snap, (10, 10), (snap.shape[1]-10, snap.shape[0]-10), (0, 0, 255), 3)
                cv2.putText(snap, "SHOPLIFTER", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(snap, f"Score:{score}  {ts}", (12, snap.shape[0]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                snap_path = os.path.join(SNAPSHOT_DIR, f"shoplifter_tid{tid}_{ts}.jpg")
                cv2.imwrite(snap_path, snap)
                print(f"[ALERT] Snapshot saved: {snap_path}")
                play_alarm()
                send_whatsapp_alert(snap_path, score, tid)

            any_shoplifting = True
            overall_status  = shoplifting_status
        elif overall_status == no_detection_status:
            overall_status = not_shoplifting_status

    sc = status_color_bad if any_shoplifting else status_color_ok
    cv2.rectangle(frame, (0, 0), (w_frame, 28), (0, 0, 0), -1)
    cv2.putText(frame, overall_status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, sc, 2)

    cv2.imshow(frame_name, frame)
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord(quit_key):
        break

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
print(f"[INFO] Output saved to {output_path}")
