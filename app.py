import streamlit as st
import os
import cv2
import supervision as sv
from ultralytics import YOLO
from time import sleep
import numpy as np
from PIL import Image
import tempfile

# Constants
MODEL_NAME = 'yolov8n.pt' 
HOME = os.getcwd()
MODEL_DEFAULT_PATH = os.path.join(HOME, 'models', MODEL_NAME)

# Helper function to load model, with caching for 60 minutes
@st.cache_resource(ttl=60*60) 
def load_model(path):
    return YOLO(path)

def main():
    st.set_page_config(page_title="AUI Security", page_icon="🧊", layout="wide")
    st.image('/home/jad-tounsi/Desktop/AUI Security/images/AUI Logo White.png', width=250)
    st.title("AI Security")
    st.text("Al Akhawayan University")
    st.markdown("An AI-powered security assistant built and deployed by Jad Tounsi El Azzoiani")

    model = load_model(MODEL_DEFAULT_PATH)
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])

    if uploaded_file is not None:
        with open("temp_input_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        video_path = "temp_input_video.mp4"

        video_info = sv.VideoInfo.from_video_path(video_path)

        # Add sliders to adjust the y-coordinates of the line
        line_start_y = st.sidebar.slider("Line Start Y-coordinate", min_value=0, max_value=video_info.height, value=video_info.height // 2)
        line_end_y = st.sidebar.slider("Line End Y-coordinate", min_value=0, max_value=video_info.height, value=video_info.height // 2)

        LINE_START, LINE_END = line_configuration(video_info, line_start_y, line_end_y)

        process_video(model, video_path, LINE_START, LINE_END)

def line_configuration(video_info, line_start_y, line_end_y):
    LINE_START = (0, line_start_y)
    LINE_END = (video_info.width, line_end_y)
    return LINE_START, LINE_END

def process_video(model, video_path, LINE_START, LINE_END):
    st.write("Processing video...")

    video_info = sv.VideoInfo.from_video_path(video_path)
    LINE_START = sv.Point(*LINE_START)
    LINE_END = sv.Point(*LINE_END)

    # Configuration for annotations
    box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.5)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.5)
    line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
    byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30)

    # Prepare for video output
    _, temp_video_path = tempfile.mkstemp(suffix='.mp4')
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_info.width, video_info.height))

    frame_placeholder = st.empty()

    confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

    def callback(frame: np.ndarray, index:int) -> np.ndarray:
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # Use the confidence_threshold variable to filter the detections
        detections = detections[detections.confidence > confidence_threshold]
    
        detections = byte_tracker.update_with_detections(detections)
    
        labels = [
            f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id
            in detections
        ]

        box_annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)
        line_zone.trigger(detections)
        line_counter_annotated_frame = line_zone_annotator.annotate(box_annotated_frame, line_counter=line_zone)

        _, buffer = cv2.imencode('.jpg', line_counter_annotated_frame)
        frame_as_bytes = buffer.tobytes()
        frame_placeholder.image(frame_as_bytes, channels="BGR", use_column_width=True, output_format="auto")
        return line_counter_annotated_frame

    sv.process_video(
        source_path = video_path,
        target_path = temp_video_path,
        callback=callback
    )
    
    st.video(temp_video_path)

if __name__ == "__main__":
    main()
