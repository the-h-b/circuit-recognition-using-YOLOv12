from ultralytics import YOLO
import cv2
import pandas as pd
import os

def detect_components(image_path, model_path, classes, conf_threshold=0.5, iou_threshold=0.45):
    """Detects components in an image using YOLOv12 with NMS."""
    model = YOLO(model_path)
    model.conf = conf_threshold
    model.iou = iou_threshold

    results = model(image_path)
    detections = []
    for r in results:
        if r.boxes is not None and r.boxes.xyxy is not None:
            for box, conf, cls in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                x1, y1, x2, y2 = map(int, box)
                class_name = classes[int(cls)]
                detections.append([x1, y1, x2, y2, conf, class_name])
    detections_df = pd.DataFrame(detections, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name'])

    return detections_df

def draw_bounding_boxes(image_path, detections):
    """Draws neater bounding boxes with better text placement to prevent overlap, thin boxes, and smaller text."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    placed_labels = []  # Store label positions to avoid overlapping
    for _, row in detections.iterrows():
        x1, y1, x2, y2, conf, cls_name = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']

        color = _get_class_color(cls_name)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA) # Thin boxes

        title = f'{cls_name} {conf:.2f}'
        (text_width, text_height), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1) # Smaller text

        text_x1 = x1
        text_y1 = y1 - text_height - 5 if y1 - text_height - 5 > 0 else y1 + text_height + 5 #Adjusted label placement
        text_x2 = x1 + text_width + 5
        text_y2 = y1

        # Shift label if it overlaps with another
        for px1, py1, px2, py2 in placed_labels:
            if text_x1 < px2 and text_x2 > px1 and text_y1 < py2 and text_y2 > py1:
                text_y1 += text_height + 10 #Adjusted spacing
                text_y2 += text_height + 10

        placed_labels.append((text_x1, text_y1, text_x2, text_y2))

        # Transparent label background
        overlay = img.copy()
        cv2.rectangle(overlay, (text_x1, text_y1), (text_x2, text_y2), color, -1, lineType=cv2.LINE_AA)
        img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

        # Draw label text with black outline
        cv2.putText(img, title, (text_x1 + 3, text_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA) #Smaller text and thinner outline
        cv2.putText(img, title, (text_x1 + 3, text_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA) #Smaller text

    return img

def _get_class_color(class_name):
    """Generates a unique color for each class."""
    color_map = {
        'Resistor': (255, 0, 0),
        'Capacitor': (0, 255, 0),
        'Inductor': (0, 0, 255),
        'Diode': (255, 255, 0),
        'Gnd': (255, 0, 255),
        'DCSource': (0, 255, 255),
        'ACSource': (128, 0, 0),
        'Ammeter': (0, 128, 0),
        'Voltmeter': (0, 0, 128),
        'NMOS': (128, 128, 0),
        'PMOS': (128, 0, 128),
        'NPN': (0, 128, 128),
        'PNP': (192, 192, 192),
        'AND': (255, 165, 0),
        'OR': (255, 20, 147),
        'NOT': (173, 216, 230),
        'XOR': (218, 112, 214),
        'NAND': (255, 99, 71),
        'Cell': (255, 215, 0),
        'DepSource': (245, 245, 220),
        'DepcurrentSrc': (230, 230, 250),
        'DCcurrentsrc': (240, 230, 140)
    }
    return color_map.get(class_name, (255, 255, 255))

def shrink_boxes(detections_df, shrink_factor=0.8):
    """Shrinks bounding boxes by a given factor."""
    adjusted_detections = []
    for _, row in detections_df.iterrows():
        xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        width = xmax - xmin
        height = ymax - ymin
        center_x = xmin + width / 2
        center_y = ymin + height / 2

        new_width = width * shrink_factor
        new_height = height * shrink_factor

        new_xmin = int(center_x - new_width / 2)
        new_ymin = int(center_y - new_height / 2)
        new_xmax = int(center_x + new_width / 2)
        new_ymax = int(center_y + new_height / 2)

        adjusted_detections.append([new_xmin, new_ymin, new_xmax, new_ymax, row['confidence'], row['name']])

    return pd.DataFrame(adjusted_detections, columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name'])