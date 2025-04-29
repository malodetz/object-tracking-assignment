def compute_iou(box1, box2):
    if len(box1) < 4 or len(box2) < 4:
        return 0.0
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    x_left = max(x1, x1b)
    y_top = max(y1, y1b)
    x_right = min(x2, x2b)
    y_bottom = min(y2, y2b)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    union_area = box1_area + box2_area - intersection_area
    if union_area == 0:
        return 0.0
    return intersection_area / union_area
