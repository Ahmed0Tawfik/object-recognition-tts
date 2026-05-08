import cv2
from distance_color_utils import estimate_distance, proximity_alert, detect_dominant_color

# Colors for alerts (BGR mapping)
ALERT_COLORS = {
    '🔴': (0,   0, 255),
    '🟠': (0, 100, 255),
    '🟡': (0, 255, 255),
    '🟢': (0, 255,   0),
}

def process_detections(model, results, bgr_image):
    """
    Given YOLO results and an image, process each detection
    for distance and color.
    """
    detections = []

    for box in results[0].boxes:
        cls_id     = int(box.cls)
        class_name = model.names[cls_id]
        confidence = float(box.conf)
        xyxy       = box.xyxy[0].cpu().numpy()
        xywh       = box.xywh[0].cpu().numpy()

        pixel_width = float(xywh[2])

        distance = estimate_distance(class_name, pixel_width)
        alert    = proximity_alert(distance)
        color    = detect_dominant_color(bgr_image, xyxy)

        detections.append({
            'class_id'   : cls_id,
            'class_name' : class_name,
            'confidence' : round(confidence, 3),
            'color'      : color,
            'distance_m' : distance,
            'alert'      : alert,
            'xyxy'       : [round(float(v), 1) for v in xyxy],
            'xywh'       : [round(float(v), 1) for v in xywh],
        })

    detections.sort(key=lambda d: d['distance_m'] if d['distance_m'] else 999)
    return detections

def annotate_image(bgr_image, detections):
    """
    Annotate detections on the image.
    """
    annotated = bgr_image.copy()

    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d['xyxy']]
        emoji  = d['alert'][0] if d['alert'] != 'distance unknown' else '🟢' # Use a default fallback
        color  = ALERT_COLORS.get(emoji, (255, 255, 255))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        dist_str = f"{d['distance_m']}m" if d['distance_m'] else 'dist?'
        label = f"{d['class_name']} | {d['color']} | {dist_str}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    return annotated
