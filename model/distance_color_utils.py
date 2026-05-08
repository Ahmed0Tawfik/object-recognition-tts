import cv2
import numpy as np

# Known real-world widths (meters) for distance calculation
REAL_WORLD_WIDTHS = {
    'person':        0.50,
    'bicycle':       0.60,
    'car':           1.80,
    'motorcycle':    0.80,
    'bus':           2.50,
    'truck':         2.50,
    'traffic light': 0.30,
    'fire hydrant':  0.30,
    'stop sign':     0.75,
    'bench':         1.50,
    'dog':           0.50,
    'cat':           0.30,
    'chair':         0.50,
    'couch':         2.00,
    'dining table':  1.50,
    'laptop':        0.35,
    'cell phone':    0.07,
    'bottle':        0.08,
    'cup':           0.08,
    'book':          0.20,
    'stairs':            1.00,
    'door':              0.90,
    'Tree':              1.00,
    'Electric_pole':     0.30,
    'Uncovered_manhole': 0.60,
    'Traffic_signs':     0.60,
}

IMAGE_WIDTH  = 640
FOCAL_LENGTH = IMAGE_WIDTH * 0.8

def estimate_distance(class_name, pixel_width, focal_length=FOCAL_LENGTH):
    """Estimate distance using pinhole camera model."""
    real_width = REAL_WORLD_WIDTHS.get(class_name)
    if real_width is None or pixel_width < 1:
        return None
    distance = (real_width * focal_length) / pixel_width
    return round(distance, 2)

def proximity_alert(distance_m):
    """Convert distance to a verbal or structural alert."""
    if distance_m is None:
        return 'distance unknown'
    if distance_m < 1.0:
        return '🔴 DANGER  — less than 1m'
    elif distance_m < 2.0:
        return '🟠 WARNING — within 2m'
    elif distance_m < 4.0:
        return '🟡 CAUTION — within 4m'
    else:
        return f'🟢 CLEAR   — {distance_m:.1f}m away'

# HSV color ranges
COLOR_RANGES = {
    'red'   : [(  0,  70,  70), ( 10, 255, 255)],
    'red2'  : [(170,  70,  70), (180, 255, 255)],
    'orange': [( 10,  70,  70), ( 25, 255, 255)],
    'yellow': [( 25,  70,  70), ( 35, 255, 255)],
    'green' : [( 35,  70,  70), ( 85, 255, 255)],
    'cyan'  : [( 85,  70,  70), (100, 255, 255)],
    'blue'  : [(100,  70,  70), (130, 255, 255)],
    'purple': [(130,  70,  70), (160, 255, 255)],
    'pink'  : [(160,  70,  70), (170, 255, 255)],
    'white' : [(  0,   0, 210), (180,  30, 255)],
    'gray'  : [(  0,   0,  80), (180,  30, 210)],
    'black' : [(  0,   0,   0), (180, 255,  80)],
}

def detect_dominant_color(bgr_image, xyxy_box):
    """Detect dominant color within a bounding box."""
    x1, y1, x2, y2 = [int(v) for v in xyxy_box]

    margin_x = int((x2 - x1) * 0.10)
    margin_y = int((y2 - y1) * 0.10)
    
    y_start = y1 + margin_y
    y_end = max(y_start + 1, y2 - margin_y)
    x_start = x1 + margin_x
    x_end = max(x_start + 1, x2 - margin_x)
    
    roi = bgr_image[y_start:y_end, x_start:x_end]

    if roi.size == 0:
        return 'unknown'

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    scores = {}

    for color_name, (lower, upper) in COLOR_RANGES.items():
        if color_name == 'red2':
            continue

        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        if color_name == 'red':
            lo2, hi2 = COLOR_RANGES['red2']
            mask2 = cv2.inRange(hsv, np.array(lo2), np.array(hi2))
            mask = cv2.bitwise_or(mask, mask2)

        scores[color_name] = int(np.count_nonzero(mask))

    if not scores:
        return 'unknown'
        
    dominant = max(scores, key=scores.get)

    total_pixels = roi.shape[0] * roi.shape[1]
    if scores[dominant] < total_pixels * 0.10:
        return 'unknown'

    return dominant
