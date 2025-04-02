import cv2
import numpy as np
import streamlit as st
import time
import os

# Global variable for detected objects
DETECTED_OBJECTS = []

# Load YOLOv3 model with updated cache decorator
@st.cache_resource
def load_model():
    # Use relative paths
    net = cv2.dnn.readNetFromDarknet(
        "yolov3.cfg",
        "yolov3.weights"
    )
    return net

net = load_model()

def get_output_layers(net):
    return net.getUnconnectedOutLayersNames()

# Load class names using relative path
@st.cache_resource
def load_classes():
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

classes = load_classes()

def detect_objects(img):
    detected_objects = []
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(get_output_layers(net))
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                label = classes[class_id]
                detected_objects.append(label)
    
    DETECTED_OBJECTS.extend(detected_objects)
    return img, detected_objects

# Streamlit UI
st.title("YOLOv3 Object Detection with History")

with st.sidebar:
    st.header("Detection History")
    if st.button("Clear Object History"):
        DETECTED_OBJECTS.clear()
    history_placeholder = st.empty()

camera_address = st.text_input("Camera Address (0/1/URL):", "0")
run = st.checkbox('Run Webcam')

frame_placeholder = st.empty()
cap = None

if run:
    try:
        if camera_address.isdigit():
            camera_address = int(camera_address)
        cap = cv2.VideoCapture(camera_address)

        if not cap.isOpened():
            st.error("Failed to initialize camera!")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                processed_frame, detected_objects = detect_objects(frame)
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, use_container_width=True)
                
                with history_placeholder.container():
                    st.write("### Detected Objects:")
                    if DETECTED_OBJECTS:
                        for obj in set(DETECTED_OBJECTS):
                            st.write(f"- {obj} (x{DETECTED_OBJECTS.count(obj)})")
                    else:
                        st.write("No objects detected yet")
                
                time.sleep(0.1)  # Add small delay to prevent high CPU usage
                
                # Check if user has unchecked the box
                if not st.session_state.get('Run Webcam', True):
                    break
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if not run:
    st.write("Camera is stopped")