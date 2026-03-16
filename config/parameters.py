WIDTH = 500

# Status labels
start_status       = "Loading..."
shoplifting_status = "!! SHOPLIFTING DETECTED !!"
not_shoplifting_status = "Normal"
no_detection_status    = "No Detection"

# Bounding box / overlay colors (BGR)
cls0_rect_color = (0, 255, 255)   # Normal person  - cyan
cls1_rect_color = (0, 0, 255)     # Suspicious     - red
conf_color      = (255, 255, 0)   # Confidence     - yellow
status_color_ok  = (0, 200, 0)    # Normal status  - green
status_color_bad = (0, 0, 255)    # Alert status   - red

frame_name = "Shoplifting Detection"
quit_key   = 'q'

# ── Suspicion score engine ────────────────────────────────────────────────────
ALERT_SCORE_THRESHOLD  = 70   # Trigger alert when score exceeds this

# Score weights
SCORE_REACH_TO_CONCEAL = 40   # Hand moves from reach zone → conceal/pocket zone
SCORE_REACH_TO_BAG     = 40   # Hand moves from reach zone → bag/hip zone
SCORE_REPEATED_REACH   = 20   # Repeated reach gestures (> threshold count)
SCORE_ABNORMAL_MOTION  = 30   # Fast erratic wrist movement

# ── Pose / tracking config ────────────────────────────────────────────────────
HAND_HISTORY_FRAMES      = 45   # Rolling window of frames for hand history
ABNORMAL_MOTION_PX       = 35   # Pixel displacement per frame = abnormal
REPEATED_REACH_COUNT     = 3    # How many reach events before scoring

# ── Body-relative zone boundaries (fraction of person bounding box height) ───
# Zones are defined top-to-bottom relative to the person's bounding box
#
#  ┌──────────────┐  y = 0   (top of person box)
#  │  HEAD ZONE   │  0.00 – 0.20
#  ├──────────────┤
#  │  REACH ZONE  │  0.20 – 0.55  (arms extended, reaching for objects)
#  ├──────────────┤
#  │ CONCEAL ZONE │  0.55 – 0.75  (pocket / waist area)
#  ├──────────────┤
#  │   BAG ZONE   │  0.75 – 1.00  (hip / bag / lower body)
#  └──────────────┘  y = 1   (bottom of person box)

REACH_ZONE_TOP   = 0.20
REACH_ZONE_BOT   = 0.55
CONCEAL_ZONE_TOP = 0.55
CONCEAL_ZONE_BOT = 0.75
BAG_ZONE_TOP     = 0.75
BAG_ZONE_BOT     = 1.00
