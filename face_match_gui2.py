import streamlit as st
import cv2
import face_recognition
import numpy as np
import tempfile
import os
from PIL import Image

# --- Helper Functions ---

def get_face_encoding(image):
    """
    Takes an image file, reads it, and returns the face encoding.
    Returns None if no face is found.
    """
    try:
        # Convert the uploaded file to an OpenCV image
        file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_img)
        
        if face_encodings:
            return face_encodings[0]
        else:
            return None
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def find_matching_frames(video_file, known_encoding):
    """
    Processes a video file to find frames matching the known face encoding.
    Returns a list of tuples (timestamp, frame).
    """
    if known_encoding is None:
        return []

    matches = []
    
    # Use a temporary file to handle the video stream
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())

    video_capture = cv2.VideoCapture(tfile.name)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        
        # Process every Nth frame to speed up processing
        if frame_count % 5 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    results = face_recognition.compare_faces([known_encoding], face_encoding)
                    if results[0]:
                        timestamp = frame_count / fps
                        # Draw a rectangle around the face
                        top, right, bottom, left = face_locations[0]
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        matches.append((timestamp, frame))
                        break # Move to the next frame once a match is found
    
    video_capture.release()
    os.unlink(tfile.name) # Clean up the temporary file
    return matches

# --- Streamlit UI ---

st.set_page_config(page_title="Face Matcher", layout="wide")

st.title("👨‍👩‍👧‍👦 Face Matcher in Videos")
st.write("Upload an image with a face and multiple videos to find where that person appears.")

# --- File Uploaders ---
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Image")
    image_file = st.file_uploader("Upload an image of the person to search for", type=["jpg", "jpeg", "png"])

with col2:
    st.header("Upload Videos")
    video_files = st.file_uploader("Upload videos to search within", type=["mp4", "mov", "avi"], accept_multiple_files=True)

# --- Processing and Display ---

if st.button("Find Matches"):
    if image_file is None:
        st.warning("Please upload an image first.")
    elif not video_files:
        st.warning("Please upload at least one video.")
    else:
        with st.spinner("Processing... This may take a while depending on video length and number of files."):
            # Get the encoding of the face in the uploaded image
            known_face_encoding = get_face_encoding(image_file)
            
            if known_face_encoding is None:
                st.error("Could not find a face in the uploaded image. Please try another one.")
            else:
                st.success("Face encoding from the image has been successfully created.")
                
                all_matches_found = False
                for video_file in video_files:
                    st.write(f"--- Processing video: `{video_file.name}` ---")
                    
                    matching_frames = find_matching_frames(video_file, known_face_encoding)
                    
                    if matching_frames:
                        all_matches_found = True
                        st.success(f"Found {len(matching_frames)} matches in `{video_file.name}`!")
                        
                        for timestamp, frame in matching_frames:
                            st.write(f"Match found at: **{int(timestamp // 60)} minutes and {int(timestamp % 60)} seconds**")
                            # Convert frame to PIL image for display
                            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                            st.image(pil_image, caption=f"Timestamp: {timestamp:.2f}s", use_column_width=True)
                    else:
                        st.info(f"No matches found in `{video_file.name}`.")

                if not all_matches_found:
                     st.balloons()
                     st.info("Finished processing all videos. No matches were found for the provided image.")

