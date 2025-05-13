import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser
import os

# Load model and label mappings
try:
    model = load_model("model.h5")
    label = np.load("labels.npy")
except Exception as e:
    st.error(f"Error loading model or labels: {e}")
    st.stop()

# Initialize MediaPipe
holistic = mp.solutions.holistic
hands = mp.solutions.hands
drawing = mp.solutions.drawing_utils
holis = holistic.Holistic()

# Streamlit UI
st.header("🎵 Emotion Based Music Recommender")

# Session state
if "run" not in st.session_state:
    st.session_state["run"] = True

# Load previous emotion (if exists)
emotion = ""
if os.path.exists("emotion.npy"):
    try:
        emotion = np.load("emotion.npy")[0]
        st.session_state["run"] = False if emotion else True
    except Exception:
        emotion = ""

# Emotion Processor Class
class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        landmarks = []

        if res.face_landmarks:
            base_x, base_y = res.face_landmarks.landmark[1].x, res.face_landmarks.landmark[1].y
            for lm in res.face_landmarks.landmark:
                landmarks.extend([lm.x - base_x, lm.y - base_y])

            for hand_landmarks, default_base_index in [
                (res.left_hand_landmarks, 8),
                (res.right_hand_landmarks, 8)
            ]:
                if hand_landmarks:
                    base_x = hand_landmarks.landmark[default_base_index].x
                    base_y = hand_landmarks.landmark[default_base_index].y
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x - base_x, lm.y - base_y])
                else:
                    landmarks.extend([0.0] * 42)

            input_data = np.array(landmarks).reshape(1, -1)
            pred_index = np.argmax(model.predict(input_data, verbose=0))
            pred_label = label[pred_index]
            np.save("emotion.npy", np.array([pred_label]))
            print("Detected Emotion:", pred_label)

            cv2.putText(frm, pred_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Draw landmarks
        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), thickness=-1, circle_radius=1),
                               connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# User inputs
lang = st.text_input("Language", placeholder="e.g., English, Hindi")
singer = st.text_input("Singer", placeholder="Optional")

# Webcam streaming
if lang and st.session_state["run"]:
    webrtc_streamer(
        key="emotion-capture",
        desired_playing_state=True,
        video_processor_factory=EmotionProcessor
    )

# Recommendation button
if st.button("🎧 Recommend me songs"):
    if not emotion:
        st.warning("⚠️ Please allow webcam access to detect your emotion first.")
        st.session_state["run"] = True
    else:
        # Generate YouTube search query
        query = f"{lang} {emotion} song {singer}".strip().replace("  ", " ")
        url = f"https://www.youtube.com/results?search_query={query}"
        webbrowser.open(url)

        # Reset state
        np.save("emotion.npy", np.array([""]))
        st.session_state["run"] = False
