import streamlit as st
import cv2
import numpy as np
from detection import detect_components, draw_bounding_boxes
from nodes import detect_wires, find_intersections, draw_nodes  
from clustering import cluster_components
from mapping import generate_netlist
from circuit_netlist_generator import generate_netlist_from_image
import tempfile
import os
from ultralytics import YOLO

# Define your classes
classes = ['ACSource', 'AND', 'Ammeter', 'Capacitor', 'Cell', 'DCSource', 'DCcurrentsrc', 'DepSource', 
           'DepcurrentSrc', 'Diode', 'Gnd', 'Inductor', 'NAND', 'NMOS', 'NOT', 'NPN', 'PMOS', 'PNP', 
           'Resistor', 'Voltmeter', 'XOR']

# Define your YOLO model path
model_path = "F:/Research/YOLOv12/runs/detect/train29/weights/best.pt"

def main():
    st.title("Hand-Drawn Circuit to LTSpice Netlist")

    uploaded_file = st.file_uploader("Upload an image of your circuit", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
            temp_image_path = temp_image.name
            cv2.imwrite(temp_image_path, img)

        st.image(img, caption="Uploaded Circuit", use_column_width=True)

        if st.button("Process Image"):
            with st.spinner("Processing..."):
                # Load YOLO model
                model = YOLO(model_path)  

                # Component Detection
                detections = detect_components(temp_image_path, model_path, classes)
                detected_image = draw_bounding_boxes(temp_image_path, detections)
                st.image(detected_image, caption="Detected Components", use_column_width=True)

                # Wire Detection
                wires_skeleton = detect_wires(temp_image_path)  # Calls function from nodes.py
                st.image(wires_skeleton, caption="Detected Wires", use_column_width=True, channels="GRAY")

                # Node Detection (Intersections & Endpoints)
                intersections = find_intersections(temp_image_path, detections)  # Calls function from nodes.py
                nodes_image = draw_nodes(temp_image_path, intersections)  # Calls function from nodes.py
                # st.image(nodes_image, caption="Detected Nodes", use_column_width=True)

                # Clustering Components with Nodes
                labels, component_labels = cluster_components(detections, intersections)

                # Netlist Generation Using YOLO and Gemini API
                netlist = generate_netlist(detections, intersections, labels, component_labels, model)  # From mapping.py
                gemini_netlist = generate_netlist_from_image(temp_image_path)  # Gemini API-based Netlist

                # st.subheader("Generated LTSpice Netlist:")
                # if netlist:
                #     st.code("\n".join(netlist), language="spice")
                # else:
                #     st.write("Could not generate netlist using detection-based approach.")

                st.subheader("Generated LTSpice Netlist ")
                if gemini_netlist:
                    st.code(gemini_netlist, language="spice")
                else:
                    st.write("Could not generate a valid netlist.")

            # Clean up temporary file
            os.unlink(temp_image_path)

if __name__ == "__main__":
    main()
