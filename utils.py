# streamlit_circuit_app/utils.py
import cv2
import numpy as np
import os

def preprocess_image(image_path):
    """Preprocesses the image for better detection."""
    img = cv2.imread(image_path)
    if img is None:
        return None #Handle invalid image path.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def resize_image(image_path, max_size=800):
    """Resizes the image to a maximum size while maintaining aspect ratio."""
    img = cv2.imread(image_path)
    if img is None:
        return None #Handle invalid image path.
    height, width = img.shape[:2]

    if max(height, width) > max_size:
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
    else:
        return img

def save_image(image, save_path):
    """Saves an OpenCV image to a file path."""
    if image is not None:
        cv2.imwrite(save_path, image)

def load_image(image_path):
    """Loads an image from a file path."""
    if os.path.exists(image_path): # Check if the file exists before attempting to load it.
        return cv2.imread(image_path)
    else:
        return None

def convert_to_grayscale(image):
    """Converts an image to grayscale."""
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return None

def apply_gaussian_blur(image, kernel_size=(5, 5), sigmaX=0):
    """Applies Gaussian blur to an image."""
    if image is not None:
        return cv2.GaussianBlur(image, kernel_size, sigmaX)
    else:
        return None

def apply_canny_edge_detection(image, threshold1=50, threshold2=150):
    """Applies Canny edge detection to an image."""
    if image is not None:
        return cv2.Canny(image, threshold1, threshold2)
    else:
        return None

def apply_threshold(image, threshold_value=127, max_value=255, threshold_type=cv2.THRESH_BINARY):
    """Applies a simple threshold to an image."""
    if image is not None:
        _, thresh = cv2.threshold(image, threshold_value, max_value, threshold_type)
        return thresh
    else:
        return None

def apply_adaptive_threshold(image, max_value=255, adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            threshold_type=cv2.THRESH_BINARY, block_size=11, C=2):
    """Applies adaptive thresholding to an image."""
    if image is not None:
        return cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, C)
    else:
        return None

def get_image_dimensions(image_path):
    """Returns the height and width of an image."""
    img = cv2.imread(image_path)
    if img is not None:
        height, width = img.shape[:2]
        return height, width
    else:
        return None, None

def draw_circle(image, center, radius, color, thickness=1):
    """Draws a circle on an image."""
    if image is not None:
        cv2.circle(image, center, radius, color, thickness)
        return image
    else:
        return None

def draw_rectangle(image, top_left, bottom_right, color, thickness=1):
    """Draws a rectangle on an image."""
    if image is not None:
        cv2.rectangle(image, top_left, bottom_right, color, thickness)
        return image
    else:
        return None

def draw_text(image, text, position, font, font_scale, color, thickness=1):
    """Draws text on an image."""
    if image is not None:
        cv2.putText(image, text, position, font, font_scale, color, thickness)
        return image
    else:
        return None

def get_image_size(image_path):
    """Returns the size of an image file in bytes."""
    if os.path.exists(image_path):
        return os.path.getsize(image_path)
    else:
        return None

def display_image(image, window_name="Image"):
    """Displays an image in a window."""
    if image is not None:
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def check_image_exists(image_path):
    """Checks if an image file exists."""
    return os.path.exists(image_path)

def get_file_extension(file_path):
    """Gets the file extension of a file."""
    return os.path.splitext(file_path)[1]

def get_filename(file_path):
    """Gets the filename without extension."""
    return os.path.splitext(os.path.basename(file_path))[0]