import math 
import cv2 
import numpy as np 
from scipy.spatial import distance
from skimage.morphology import skeletonize
from intersections import segmented_intersections

def detect_wires(image_path):
    """
    Detects wires using Canny edge detection and Hough Transform.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def node_detector(gray):
    """
    Finds coordinates of junctions or nodes present in components removed image.
    Uses Hough Transform to detect lines and finds their intersections as nodes.
    """
    img = cv2.GaussianBlur(gray, (9,9), 0)
    th = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    edged = th.astype(np.uint8)
    blnk = np.zeros_like(gray).astype(np.uint8)
    
    # Drawing contours on a blank image
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blnk = cv2.drawContours(blnk, contours, -1, (255, 0, 0), 3)
    blnk = blnk == 255
    blnk = blnk.astype(np.uint8)
    
    rho, theta, threshold = 1, np.pi / 180, 15
    min_line_length, max_line_gap = 10, 15
    lines = cv2.HoughLinesP(blnk, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    
    lines_x, lines_y = [], []
    for line in lines:
        orientation = math.atan2((line[0][1] - line[0][3]), (line[0][0] - line[0][2]))
        if 45 < abs(math.degrees(orientation)) < 135:
            lines_y.append(line)
        else:
            lines_x.append(line)
    
    lines_x = sorted(lines_x, key=lambda l: l[0][0])
    lines_y = sorted(lines_y, key=lambda l: l[0][1])
    
    intersections = segmented_intersections([lines_x, lines_y])
    node_dim = []
    for i in intersections:
        for x, y in i:
            node_dim.append([y, x])
    
    return node_dim

def find_intersections(image_path, detections):
    """
    Detects wire intersections and endpoints using improved node detection.
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    nodes = node_detector(gray)
    
    # Detect component terminals
    terminals = []
    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        terminals.extend([(x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2), ((x1 + x2) // 2, y1), ((x1 + x2) // 2, y2)])
    
    # Merge nearby nodes
    all_nodes = nodes + terminals
    merged_nodes = []
    node_flags = [False] * len(all_nodes)
    
    for i in range(len(all_nodes)):
        if not node_flags[i]:
            merged_nodes.append(all_nodes[i])
            node_flags[i] = True
            for j in range(i + 1, len(all_nodes)):
                if not node_flags[j] and distance.euclidean(all_nodes[i], all_nodes[j]) < 10:
                    node_flags[j] = True
    
    return merged_nodes

def draw_nodes(image_path, nodes):
    """Draws detected nodes on the image."""
    img = cv2.imread(image_path)
    for x, y in nodes:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    return img