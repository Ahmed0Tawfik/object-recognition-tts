# ══════════════════════════════════════════════════════════
# main.py  —  Still image blind-assistance pipeline
# YOLO → distance → color → 9-position → TTS
# ══════════════════════════════════════════════════════════
import cv2
import time
import sys
from ultralytics import YOLO

from distance_color_pipeline import process_detections, annotate_image
from position_mapper          import get_position, draw_grid
from tts_speaker              import TTSSpeaker


# ── Config ─────────────────────────────────────────────────
MODEL_PATH      = 'final.pt'
IMAGE_PATH      = 'tst_img.webp'     
OUTPUT_PATH     = 'result.jpg'   
CONF_THRESH     = 0.25
IOU_THRESH      = 0.45
ANNOUNCE_DIST_M = 10.0           # only announce if closer than this
SHOW_GRID       = True           # draw 3x3 grid on result image


def build_tts_message(det, position_phrase):
    """
    Builds the spoken message for one detection.

    Example:
      "There is a car, colored blue,
       3.2 meters away, directly ahead of you."
    """
    obj   = det['class_name']
    color = det['color']
    dist  = det['distance_m']

    # Distance phrasing
    if dist is not None:
        if dist < 1.0:
            dist_phrase = "less than one meter away"
        elif dist < 2.0:
            dist_phrase = f"about {dist:.1f} meters away"
        else:
            dist_phrase = f"{dist:.1f} meters away"
    else:
        dist_phrase = "at an unknown distance"

    # Color phrasing (skip if unknown)
    color_phrase = f"colored {color}, " \
                   if color != 'unknown' else ""

    return (f"There is a {obj}, "
            f"{color_phrase}"
            f"{dist_phrase}, "
            f"{position_phrase}.")


def main():
    # ── Load image ──────────────────────────────────────────
    image_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: cannot read image at '{image_path}'")
        return

    frame_h, frame_w = frame.shape[:2]
    print(f"Image loaded: {image_path}  ({frame_w}×{frame_h})")

    # ── Load model ──────────────────────────────────────────
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    # ── Initialize TTS ──────────────────────────────────────
    print("Initializing TTS engine...")
    speaker = TTSSpeaker(rate=145, volume=1.0, cooldown_sec=0)
    # cooldown_sec=0 → no cooldown for still image
    # (we want every detection spoken, not suppressed)

    # ── YOLO inference ──────────────────────────────────────
    print("Running detection...")
    results    = model.predict(frame,
                               conf=CONF_THRESH,
                               iou=IOU_THRESH,
                               verbose=False)
    detections = process_detections(model, results, frame)

    if not detections:
        print("No objects detected.")
        speaker.speak("No objects detected in the image.")
        time.sleep(3)
        speaker.stop()
        return

    # ── Sort by distance (closest first) ────────────────────
    detections.sort(
        key=lambda d: d['distance_m'] if d['distance_m'] else 999
    )

    # ── Annotate image ──────────────────────────────────────
    annotated = annotate_image(frame.copy(), detections)
    if SHOW_GRID:
        annotated = draw_grid(annotated)

    # ── Process each detection ──────────────────────────────
    print(f"\nFound {len(detections)} object(s):\n")
    print(f"{'CLASS':<22} {'COLOR':<10} {'DIST':>8}  "
          f"{'POSITION':<25}  MESSAGE")
    print("─" * 100)

    messages = []

    for det in detections:
        # Skip objects too far away
        dist = det.get('distance_m')
        if dist is not None and dist > ANNOUNCE_DIST_M:
            continue

        # ── Get 9-position ───────────────────────────────
        cx, cy   = det['xywh'][0], det['xywh'][1]
        phrase, label, cell = get_position(
            cx, cy, frame_w, frame_h
        )

        # ── Build message ────────────────────────────────
        msg = build_tts_message(det, phrase)
        messages.append(msg)

        # ── Draw position label on image ─────────────────
        x1 = int(det['xyxy'][0])
        y1 = int(det['xyxy'][1])
        cv2.putText(
            annotated, label,
            (x1, y1 - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 0), 2
        )

        # ── Print to console ─────────────────────────────
        dist_str = f"{dist:.1f}m" if dist else "unknown"
        print(f"{det['class_name']:<22} "
              f"{det['color']:<10} "
              f"{dist_str:>8}  "
              f"{label:<25}  "
              f"{msg}")

    print("─" * 100)

    # ── Speak all messages sequentially ─────────────────────
    print(f"\nSpeaking {len(messages)} message(s)...")
    for msg in messages:
        speaker.speak(msg, key=None)   # no cooldown on still image

    # ── Show & save result ───────────────────────────────────
    cv2.imwrite(OUTPUT_PATH, annotated)
    print(f"\nAnnotated image saved → {OUTPUT_PATH}")

    cv2.imshow("Blind Assistance — Still Image", annotated)
    print("Press any key to close the window...")
    cv2.waitKey(0)   # wait for keypress (not 1ms like video)

    # ── Wait for TTS to finish before exiting ───────────────
    # Estimate: ~1.5 seconds per message
    # ✅ AFTER — blocks until every message is actually done
    print("Speaking... waiting for all messages to finish.")
    speaker.wait_until_done()
    speaker.stop()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()