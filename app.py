# streamlit_webrtc_unified.py
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from math import hypot
import threading
import pygame
import time
import os
from queue import Queue

# ----------------------------
# Audio setup
# ----------------------------
pygame_inited = False
try:
    pygame.mixer.init()
    pygame_inited = True
except:
    st.warning("Pygame mixer init failed")

audio_queue = Queue()
audio_lock = threading.Lock() if pygame_inited else None

def audio_worker():
    while True:
        path = audio_queue.get()
        if path is None:
            audio_queue.task_done()
            break
        if path and os.path.exists(path):
            try:
                with audio_lock:
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.stop()
                    pygame.mixer.music.load(path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.05)
            except Exception as e:
                print(f"[AUDIO ERROR] {e}")
        audio_queue.task_done()

if pygame_inited:
    threading.Thread(target=audio_worker, daemon=True).start()

def enqueue_audio(path):
    if pygame_inited:
        audio_queue.put(path)

# ----------------------------
# Mediapipe setup
# ----------------------------
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ----------------------------
# Helpers
# ----------------------------
FINGER_TIPS = [8, 12, 16, 20]

def fingers_up(hand_landmarks):
    fingers = [1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0]
    for tip in FINGER_TIPS:
        try:
            fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y else 0)
        except:
            fingers.append(0)
    return fingers

def detect_hand_gesture(fingers):
    gestures = {
        (0,1,0,0,0): "Bring Water",
        (0,1,1,0,0): "Emergency",
        (0,0,0,0,0): "Stop",
        (0,1,1,1,0): "Assist me outside",
        (0,1,1,1,1): "Call 108",
        (0,0,1,1,1): "Contact my caregiver",
        (0,0,0,0,1): "Check my supplies",
        (0,0,0,1,1): "Help me sit",
    }
    return gestures.get(tuple(fingers))

# ----------------------------
# Video Transformer
# ----------------------------
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Hand detection
        hand_result = hands.process(rgb)
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = fingers_up(hand_landmarks)
                gesture = detect_hand_gesture(fingers)
                if gesture:
                    cv2.putText(img, f"Gesture: {gesture}", (10,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    # enqueue_audio(GESTURE_SOUNDS.get(gesture))

        # Face mesh detection (optional: add blink/head pose here)
        face_result = face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                mp_draw.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        return img

# ----------------------------
# Streamlit WebRTC streamer
# ----------------------------
st.title("🧠 Unified Real-time Detection")
webrtc_streamer(
    key="example",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
