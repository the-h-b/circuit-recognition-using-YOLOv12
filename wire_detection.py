import cv2
import numpy as np

def detect_wires(image_path):
    """Detect wires in a circuit using edge detection & Hough Transform."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)
    
    wire_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(wire_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return wire_image, lines

def find_intersections(wire_segments, detections):
    """
    Detect intersections where wires meet and component terminals.

    Args:
        wire_segments (list): List of detected wire line segments.
        detections (list): List of detected components with bounding boxes.

    Returns:
        tuple: (intersections, terminals)
    """
    intersections = []
    terminals = []

    # Convert wire segments to a NumPy array for processing
    lines_array = np.array(wire_segments)

    # Loop through each pair of lines to check for intersections
    for i in range(len(lines_array)):
        for j in range(i + 1, len(lines_array)):
            line1 = lines_array[i]
            line2 = lines_array[j]

            # Find intersection points
            intersection = line_intersection(line1, line2)
            if intersection is not None:
                intersections.append(intersection)

    # Find wire endpoints and match them with components
    for line in wire_segments:
        x1, y1, x2, y2 = line
        terminals.append((x1, y1))
        terminals.append((x2, y2))

    # Check if terminals are near component bounding boxes
    for component in detections:
        x, y, w, h, label = component  # Assuming YOLO returns (x, y, width, height, label)
        component_center = (x + w // 2, y + h // 2)

        for term in terminals:
            if np.linalg.norm(np.array(term) - np.array(component_center)) < 15:  # Adjust threshold as needed
                terminals.append(term)

    return intersections, terminals

def line_intersection(line1, line2):
    """
    Finds the intersection point of two lines if they intersect.

    Args:
        line1 (tuple): (x1, y1, x2, y2) for line 1.
        line2 (tuple): (x1, y1, x2, y2) for line 2.

    Returns:
        tuple or None: (x, y) if intersection exists, else None.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Compute the determinant
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det == 0:
        return None  # Lines are parallel

    # Compute intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det

    # Check if the intersection point is within the segment bounds
    if min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2) and \
       min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4):
        return int(px), int(py)

    return None


def draw_nodes(image_path, nodes):
    """Draw detected nodes (intersections) on the circuit."""
    img = cv2.imread(image_path)
    for (x, y) in nodes:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Red nodes
    return img
