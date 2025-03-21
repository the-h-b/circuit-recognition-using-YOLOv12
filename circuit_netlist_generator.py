import json
from inference_sdk import InferenceHTTPClient
import streamlit as st
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="AIzaSyBnd8T1ys6AEi29YzJUKS_NIlr8N1I0YNU")

# Component classes
classes = ['ACSource', 'AND', 'Ammeter', 'Capacitor', 'Cell', 'DCSource', 'DCcurrentsrc', 'DepSource', 'DepcurrentSrc', 
           'Diode', 'Gnd', 'Inductor', 'NAND', 'NMOS', 'NOT', 'NPN', 'PMOS', 'PNP', 'Resistor', 'Voltmeter', 'XOR']

# Initialize Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="ok70J383soqUqe5LiCUN"
)

def get_yolo_results(image_path):
    """Detects components in an image using YOLO."""
    try:
        result = CLIENT.infer(image_path, model_id="yolov12compdet/1")
        return result['predictions']
    except Exception as e:
        print(f"Error calling Roboflow API: {e}")
        return None

def generate_netlist(circuit_description, model="gemini-1.5-flash"):
    """
    Generates a netlist using Gemini with a structured prompt.
    """
    try:
        model = genai.GenerativeModel(model)
        prompt = f"""
        Generate a SPICE netlist based on the following component list and wire connections.

        Components detected:
        {circuit_description}

        Rules:
        1. Connect components based on spatial proximity (within 20px)
        2. Add realistic values (resistors: 1k-100k, caps: 1n-100u)
        3. Include .tran 1ms simulation command
        4. Output ONLY the netlist code
        """

        response = model.generate_content(prompt)
        netlist = response.text.strip()

        return netlist
    except Exception as e:
        st.error(f"Error generating netlist: {e}")
        return None

def process_yolo_results(yolo_results):
    """Processes YOLO results and creates a circuit description."""
    if not yolo_results:
        return None

    circuit_description = ""
    for detection in yolo_results:
        class_name = detection.get("class")
        if class_name in classes:
            circuit_description += f"A {class_name} component. "
        else:
            print(f"Warning: Unknown class detected: {class_name}")

    return circuit_description

def generate_netlist_from_image(image_path):
    """Generates a netlist from an image."""
    yolo_results = get_yolo_results(image_path)
    if not yolo_results:
        return None

    circuit_description = process_yolo_results(yolo_results)
    if not circuit_description:
        return None

    netlist_string = generate_netlist(circuit_description)
    return netlist_string
