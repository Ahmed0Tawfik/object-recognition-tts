# ══════════════════════════════════════════════════════════
# position_mapper.py
# Maps bounding box center → one of 9 screen regions,
# then converts to a natural spoken phrase.
#
#   ┌─────────────┬──────────────┬─────────────┐
#   │  top left   │  top center  │  top right  │
#   ├─────────────┼──────────────┼─────────────┤
#   │    left     │ ahead of you │    right    │
#   ├─────────────┼──────────────┼─────────────┤
#   │ bottom left │bottom center │bottom right │
#   └─────────────┴──────────────┴─────────────┘
# ══════════════════════════════════════════════════════════

# 9 region labels — (row, col) → spoken phrase
POSITION_PHRASES = {
    (0, 0): "at the top left",
    (0, 1): "at the top center",
    (0, 2): "at the top right",
    (1, 0): "to your left",
    (1, 1): "directly ahead of you",
    (1, 2): "to your right",
    (2, 0): "at the bottom left",
    (2, 1): "at the bottom center",
    (2, 2): "at the bottom right",
}

# Short labels for on-screen annotation
POSITION_LABELS = {
    (0, 0): "top-left",
    (0, 1): "top-center",
    (0, 2): "top-right",
    (1, 0): "left",
    (1, 1): "center",
    (1, 2): "right",
    (2, 0): "bot-left",
    (2, 1): "bot-center",
    (2, 2): "bot-right",
}


def get_position(cx, cy, frame_w, frame_h):
    """
    Returns (spoken_phrase, short_label, grid_cell).

    cx, cy      : bounding box center in pixels
    frame_w/h   : frame dimensions in pixels
    """
    # Normalize to 0–1
    rel_x = cx / frame_w
    rel_y = cy / frame_h

    # Map to grid column (0=left, 1=center, 2=right)
    col = 0 if rel_x < 1/3 else (1 if rel_x < 2/3 else 2)

    # Map to grid row    (0=top,  1=middle, 2=bottom)
    row = 0 if rel_y < 1/3 else (1 if rel_y < 2/3 else 2)

    cell    = (row, col)
    phrase  = POSITION_PHRASES[cell]
    label   = POSITION_LABELS[cell]

    return phrase, label, cell


def draw_grid(frame):
    """
    Draws the 3x3 grid overlay on the frame.
    Useful for debugging / demo purposes.
    """
    import cv2
    h, w = frame.shape[:2]

    grid_color = (80, 80, 80)   # dark gray

    # Vertical dividers
    cv2.line(frame, (w//3, 0),   (w//3, h),   grid_color, 1)
    cv2.line(frame, (2*w//3, 0), (2*w//3, h), grid_color, 1)

    # Horizontal dividers
    cv2.line(frame, (0, h//3),   (w, h//3),   grid_color, 1)
    cv2.line(frame, (0, 2*h//3), (w, 2*h//3), grid_color, 1)

    # Region labels (faint)
    for (row, col), lbl in POSITION_LABELS.items():
        tx = int((col + 0.5) * w / 3)
        ty = int((row + 0.5) * h / 3)
        cv2.putText(
            frame, lbl, (tx - 35, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (100, 100, 100), 1
        )
    return frame