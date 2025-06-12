import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import datetime
import pandas as pd
from insightface.app import FaceAnalysis

# Initialize InsightFace model
@st.cache_resource
def load_model():
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    return app

app = load_model()

st.title("🎯 Improved Face Matching in CCTV Video")

# Upload face image and video
query_img_file = st.file_uploader("Upload Query Face Image", type=["jpg", "jpeg", "png"])
video_file = st.file_uploader("Upload CCTV Video File", type=["mp4", "avi", "mov"])

# Set similarity threshold
similarity_threshold = st.slider("Set Similarity Threshold", 0.2, 0.6, 0.28, 0.01)

# Process on file upload
if query_img_file and video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_img.write(query_img_file.read())
        tmp_vid.write(video_file.read())
        query_img_path = tmp_img.name
        video_path = tmp_vid.name

    # Extract embedding from query image (centered on the largest face)
    query_img = cv2.imread(query_img_path)
    query_faces = app.get(query_img)
    if not query_faces:
        st.error("❌ No face detected in the uploaded query image.")
        st.stop()

    # Use the largest face
    query_face = max(query_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    query_embedding = query_face.embedding

    # Prepare to process video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_dir = tempfile.mkdtemp()
    matches = []
    frame_num = 0

    # Show progress
    stframe = st.empty()
    progress = st.progress(0)

    def is_blurry(image, threshold=80):
        return cv2.Laplacian(image, cv2.CV_64F).var() < threshold

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        

        if frame_num % 5 != 0:
            frame_num += 1
            continue

        faces = app.get(frame)
        for face in faces:

            box = face.bbox.astype(int)
            face_crop = frame[box[1]:box[3], box[0]:box[2]]
            
            if face_crop.size>0 and is_blurry(face_crop):
                blur = 1                
            else:
                blur = 0

            print(f"Processing frame {frame_num}/{total_frames}, face_crop size {face_crop.size}, is_blurry {blur}")
            # if face_crop.size == 0 or is_blurry(face_crop):
               # continue
               # print(f"Processing frame {frame_num}/{total_frames}, face_crop size {face_crop.size}, is_blurry {blur}")

            # Normalize embeddings before cosine similarity
            query_emb_norm = query_embedding / np.linalg.norm(query_embedding)
            face_emb_norm = face.embedding / np.linalg.norm(face.embedding)
            sim = np.dot(query_emb_norm, face_emb_norm)
            print(f"Processing frame {frame_num}/{total_frames}, sim {sim} threshold {similarity_threshold}")
            if sim > similarity_threshold:
                timestamp = str(datetime.timedelta(seconds=frame_num / fps))
                filename = os.path.join(output_dir, f"match_{frame_num}_sim_{sim:.2f}.jpg")
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"sim: {sim:.2f}", (box[0], box[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite(filename, frame)
                matches.append((frame_num, timestamp, sim, filename))

        frame_num += 1
        progress.progress(min(frame_num / total_frames, 1.0))

    cap.release()

    # Display results
    if matches:
        df = pd.DataFrame(matches, columns=["Frame", "Timestamp", "Similarity", "Image Path"])
        st.success(f"✅ Found {len(matches)} matching frames")
        st.dataframe(df.drop(columns=["Image Path"]))

        st.markdown("### 🖼️ Matched Frames")
        for _, row in df.iterrows():
            st.image(row["Image Path"], caption=f"Frame {row['Frame']} - Sim: {row['Similarity']:.2f} at {row['Timestamp']}")
    else:
        st.warning("No matches found in the video with the current threshold.")
