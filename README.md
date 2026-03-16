# Shoplifting Detection System

Real-time shoplifting detection using YOLOv8-Pose. Detects suspicious body movements via keypoint analysis and sends a WhatsApp alert with a snapshot photo.

---

## How It Works

1. Camera feed (video file or RTSP) is read frame by frame
2. YOLOv8-Pose detects people and their body keypoints (wrists, elbows, shoulders)
3. Each person is tracked across frames using IoU matching
4. Wrist positions are mapped to body zones (reach / conceal / bag)
5. A suspicion score is calculated based on movement patterns
6. If score >= 60, an alarm sounds and a WhatsApp alert is sent with a snapshot

---

## Suspicion Scoring

| Behaviour | Score |
|---|---|
| Hand moves reach → conceal zone | +40 |
| Hand moves reach → bag zone | +40 |
| Repeated reach gestures (3+) | +20 |
| Abnormal fast wrist motion | +30 |
| Both hands in suspicious zones | +20 |
| Loitering in same spot (3+ sec) | +15 |

Alert triggers at score **60/100**

---

## Body Zones (relative to person height)

```
┌──────────────┐  0%   top of person
│  HEAD ZONE   │
├──────────────┤  22%
│  REACH ZONE  │  ← arms reaching for items
├──────────────┤  50%
│ CONCEAL ZONE │  ← pocket / waist area
├──────────────┤  72%
│   BAG ZONE   │  ← hip / bag area
└──────────────┘  100% bottom of person
```

---

## Project Structure

```
├── shoplifting_detection.py   # main script
├── alert.py                   # WhatsApp alert + Cloudinary upload
├── config/
│   ├── parameters.py          # all tunable settings
│   ├── alerts.py              # loads credentials from .env
│   └── alerts.example.py      # template — copy to alerts.py
├── .env                       # credentials (never commit this)
├── Samples/
│   ├── inputs/                # test video files
│   └── outputs/
│       └── snapshots/         # saved alert snapshots
├── yolov8n-pose.pt            # YOLOv8 pose model
└── requirments.txt
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirments.txt
```

### 2. Configure credentials
Copy the example and fill in your values:
```bash
copy config\alerts.example.py config\alerts.py
```

Or create a `.env` file:
```
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_FROM=whatsapp:+14155238886
OWNER_WHATSAPP=whatsapp:+your_number
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_UPLOAD_PRESET=your_preset
```

### 3. Set video source
In `shoplifting_detection.py`, line ~50:
```python
# For test video:
cap = cv2.VideoCapture("Samples/inputs/sr1.mp4")

# For live RTSP camera:
cap = cv2.VideoCapture("rtsp://admin:PASSWORD@CAMERA_IP:554/ch1/main")
```

### 4. Run
```bash
python shoplifting_detection.py
```

---

## Services Required

### Twilio (WhatsApp alerts)
1. Sign up at https://twilio.com
2. Enable WhatsApp sandbox at https://console.twilio.com/us1/develop/sms/try-it-out/whatsapp-learn
3. Send the join code from your phone to activate sandbox
4. Copy Account SID and Auth Token to `.env`

### Cloudinary (image hosting for WhatsApp)
1. Sign up free at https://cloudinary.com
2. Dashboard → Settings → Upload → Upload Presets → Add preset
3. Set signing mode to **Unsigned**
4. Copy Cloud Name and Preset Name to `.env`

---

## Tuning Detection

All settings are in `config/parameters.py`:

| Setting | Default | Effect |
|---|---|---|
| `ALERT_SCORE_THRESHOLD` | 60 | Lower = more sensitive, higher = fewer false alarms |
| `REPEATED_REACH_COUNT` | 3 | How many reaches before scoring |
| `REACH_DWELL_FRAMES` | 5 | Frames wrist must stay in zone to count |
| `TRACK_GRACE_FRAMES` | 20 | Frames to keep tracking after person disappears |
| `ZONE_SMOOTH_FRAMES` | 4 | Smoothing window to reduce jitter |

---

## Controls

| Key | Action |
|---|---|
| `q` | Quit |

---

## Requirements

- Python 3.8+
- Windows (for `winsound` alarm — remove it on Linux/Mac)
- Internet connection for Twilio + Cloudinary alerts
- RTSP camera or video file
