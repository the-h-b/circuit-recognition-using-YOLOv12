import torch
import cv2
import numpy as np

def detect_components(image_path, model_path, classes):
    """Detects components in an image using YOLOv12."""
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    model.classes = classes
    model.conf = 0.5  # Confidence threshold

    results = model(image_path)
    detections = results.pandas().xyxy[0]  # Pandas DataFrame with detections

    return detections

def draw_bounding_boxes(image_path, detections):
    """Draws bounding boxes on the image."""
    img = cv2.imread(image_path)
    for _, row in detections.iterrows():
        x1, y1, x2, y2, conf, cls_name = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['name']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{cls_name} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    # Example usage (for testing)
    classes = ['ACSource', 'AND', 'Ammeter', 'Capacitor', 'Cell', 'DCSource', 'DCcurrentsrc', 'DepSource', 'DepcurrentSrc', 'Diode', 'Gnd', 'Inductor', 'NAND', 'NMOS', 'NOT', 'NPN', 'PMOS', 'PNP', 'Resistor', 'Voltmeter', 'XOR']
    image_path = "F:\yolov12ckt\dataset\test\images\Screenshot-2025-03-01-163155_png.rf.e38491f4b0b2a789abc0619245259b6b.jpg"
    model_path = "F:\yolov12ckt\runs\detect\train29\weights\best.pt"
    detections = detect_components(image_path, model_path, classes)
    output_image = draw_bounding_boxes(image_path, detections)
    cv2.imshow("Detected Components", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()