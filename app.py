import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import threading
import pygame
import time
import os
from queue import Queue

# =========================================================
# 1. AUDIO SETUP
# =========================================================
# --- IMPORTANT ---
# Replace these with relative paths (e.g., "Audios/bring_water.mp3") 
# or comment out if deploying to Streamlit Cloud without files.
# ---

# Hand Gestures (from hand.py)
HAND_GESTURE_SOUNDS = {
    "Bring Water": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\bring water.mp3",
    "Emergency": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\emergency.mp3",
    # ... Add all other paths if you want them ...
}
# Head Poses (from head.py)
HEAD_POSE_SOUNDS = {
    "Up": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\bring_water.opus",
    "Down": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\emergency1.opus",
    "Right": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\hungry.opus",
    "Left": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\i_need_firstaid.opus"
}
# Blink Events (from eyeblink.py)
BLINK_SOUNDS = {
    "Single Blink": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\call_for_caretaker.opus",
    "Double Blink": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\need_help.opus"
}

pygame_inited = False
try:
    pygame.mixer.init()
    pygame_inited = True
except Exception as e:
    st.warning(f"Pygame mixer init failed: {e}")

audio_queue = Queue()
audio_lock = threading.Lock() if pygame_inited else None

def audio_worker():
    """Plays audio files from the queue sequentially."""
    while True:
        path = audio_queue.get()
        if path is None:
            audio_queue.task_done()
            break
        if path and os.path.exists(path):
            try:
                if audio_lock:
                    with audio_lock:
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.stop()
                        pygame.mixer.music.load(path)
                        pygame.mixer.music.play()
                        # Wait for playback to finish
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.05)
            except Exception as e:
                print(f"[AUDIO ERROR] {e}")
        audio_queue.task_done()

if pygame_inited:
    threading.Thread(target=audio_worker, daemon=True).start()

def enqueue_audio(path):
    """Adds an audio path to the playback queue."""
    if pygame_inited and path:
        audio_queue.put(path)

# =========================================================
# 2. MEDIAPIPE & CONSTANTS SETUP
# =========================================================
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6)

# Head Pose Constants (from head.py)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), # Nose, Left Eye, Right Eye
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0), (0.0, -330.0, -65.0) # Left Mouth, Right Mouth, Chin
])
FACE_LANDMARK_POINTS = [1, 33, 263, 61, 291, 152] # Nose, Left Eye, Right Eye, Left Mouth, Right Mouth, Chin
HEAD_YAW_THRESHOLD = 10
HEAD_PITCH_THRESHOLD = 12
HEAD_STABLE_TIME = 0.5

# Blink Constants (from eyeblink.py)
LEFT_EYE_EAR_INDICES = [362, 380, 374, 263, 386, 385]
RIGHT_EYE_EAR_INDICES = [33, 159, 158, 133, 153, 145]
EAR_THRESHOLD = 0.3
CONSEC_FRAMES = 4

# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

# --- Hand Gesture Helpers (from hand.py) ---
FINGER_TIPS = [8, 12, 16, 20]
def fingers_up(hand_landmarks):
    # This logic from hand.py is for the left hand facing the camera (thumb x < x-3)
    fingers = [1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0]
    for tip in FINGER_TIPS:
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)
    return fingers

def detect_gesture(fingers):
    gestures = {
        (0, 1, 0, 0, 0): "Bring Water",
        (0, 1, 1, 0, 0): "Emergency",
        (0, 0, 0, 0, 0): "Stop",
        (0, 1, 1, 1, 0): "Assist me outside",
        (0, 1, 1, 1, 1): "Call 108",
        (0, 0, 1, 1, 1): "Contact my caregiver",
        (0, 0, 0, 0, 1): "Check my supplies",
        (0, 0, 0, 1, 1): "Help me sit"
    }
    return gestures.get(tuple(fingers))

# --- Head Pose Helpers (from head.py) ---
def get_head_pose(landmarks, img_w, img_h):
    image_points = np.array([
        (landmarks[idx].x * img_w, landmarks[idx].y * img_h) for idx in FACE_LANDMARK_POINTS
    ], dtype="double")

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    # SolvePnP expects model_points and image_points to be of type double
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs
    )

    if not success:
        return None, None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, _ = eulerAngles.flatten()[:3] # Yaw (Y-axis rotation), Pitch (X-axis rotation)
    return yaw, pitch

def get_head_direction(yaw, pitch):
    if yaw > HEAD_YAW_THRESHOLD:
        return "Right"
    elif yaw < -HEAD_YAW_THRESHOLD:
        return "Left"
    elif pitch > HEAD_PITCH_THRESHOLD:
        return "Up"
    elif pitch < -HEAD_PITCH_THRESHOLD:
        return "Down"
    else:
        return "Center"

def get_head_instruction(direction):
    if direction == "Up": return "WATER"
    if direction == "Down": return "EMERGENCY"
    if direction == "Left": return "FIRST AID"
    if direction == "Right": return "FOOD"
    return "Looking straight"

# --- Blink Detection Helpers (from eyeblink.py) ---
def eye_aspect_ratio(eye_landmarks_indices, face_landmarks):
    # Convert landmarks to pixel coordinates
    def get_coords(idx):
        return np.array([face_landmarks[idx].x, face_landmarks[idx].y])

    # A, B are vertical distances
    A = np.linalg.norm(get_coords(eye_landmarks_indices[1]) - get_coords(eye_landmarks_indices[5]))
    B = np.linalg.norm(get_coords(eye_landmarks_indices[2]) - get_coords(eye_landmarks_indices[4]))
    # C is horizontal distance
    C = np.linalg.norm(get_coords(eye_landmarks_indices[0]) - get_coords(eye_landmarks_indices[3]))
    
    return (A + B) / (2.0 * C) if C != 0 else 0.0

# =========================================================
# 4. VIDEO TRANSFORMER CLASS (The Core Logic)
# =========================================================
class UnifiedVideoTransformer(VideoTransformerBase):
    def __init__(self):
        # State for Head Pose Stabilization (from head.py)
        self.stable_direction = "Center"
        self.last_detected_direction = "Center"
        self.direction_start_time = time.time()
        
        # State for Blink Counting (from eyeblink.py)
        self.blink_counter = 0
        self.double_blink_counter = 0
        self.single_blink_counter = 0
        self.last_event = None
        self.last_event_time = 0.0
        self.event_display_duration = 1.5
        
        # State for Hand Gesture (from hand.py)
        self.last_gesture = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_h, img_w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- HAND GESTURE DETECTION ---
        hand_result = hands.process(rgb)
        current_gesture = None
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = fingers_up(hand_landmarks)
                current_gesture = detect_gesture(fingers)
                
        # Audio/Display for Hand Gesture
        if current_gesture and current_gesture != self.last_gesture:
            enqueue_audio(HAND_GESTURE_SOUNDS.get(current_gesture))
            self.last_gesture = current_gesture
        elif not current_gesture:
            self.last_gesture = None
            
        if current_gesture:
            cv2.putText(img, f"Hand: {current_gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # --- FACE MESH, HEAD POSE, AND BLINK DETECTION ---
        face_result = face_mesh.process(rgb)
        
        if face_result.multi_face_landmarks:
            landmarks = face_result.multi_face_landmarks[0].landmark
            
            # Draw Face Mesh (Optional, use FACEMESH_CONTOURS for less clutter)
            mp_draw.draw_landmarks(img, face_result.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS)
            
            # 1. HEAD POSE LOGIC (Rotation)
            yaw, pitch = get_head_pose(landmarks, img_w, img_h)
            if yaw is not None:
                detected_direction = get_head_direction(yaw, pitch)

                # Stabilization Logic
                current_time = time.time()
                if detected_direction != self.last_detected_direction:
                    self.last_detected_direction = detected_direction
                    self.direction_start_time = current_time
                else:
                    if current_time - self.direction_start_time > HEAD_STABLE_TIME:
                        if self.stable_direction != detected_direction:
                            self.stable_direction = detected_direction
                            
                            # Play audio once for new stable direction
                            if self.stable_direction in HEAD_POSE_SOUNDS:
                                enqueue_audio(HEAD_POSE_SOUNDS.get(self.stable_direction))
                            elif self.stable_direction == "Center":
                                pass # Optionally stop audio here if needed

                # Display Instruction Text
                instruction_text = get_head_instruction(self.stable_direction)
                cv2.putText(img, f"Head: {instruction_text}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            
            # 2. BLINK DETECTION LOGIC
            right_ear = eye_aspect_ratio(RIGHT_EYE_EAR_INDICES, landmarks)
            left_ear = eye_aspect_ratio(LEFT_EYE_EAR_INDICES, landmarks)
            avg_ear = (right_ear + left_ear) / 2.0
            
            event_detected = None
            
            if avg_ear < EAR_THRESHOLD:
                self.blink_counter += 1
            else:
                if self.blink_counter >= CONSEC_FRAMES:
                    # Blink occurred (single or double)
                    current_time_for_blink = time.time()
                    
                    if self.last_event == "Single Blink" and (current_time_for_blink - self.last_event_time) < 0.8: # 0.8s max interval
                        # DOUBLE BLINK
                        self.double_blink_counter += 1
                        event_detected = "Double Blink"
                        enqueue_audio(BLINK_SOUNDS.get("Double Blink"))
                        self.last_event = None # Reset for next cycle
                    elif self.last_event == "Single Blink" and (current_time_for_blink - self.last_event_time) >= 0.8:
                        # Single blink was completed, but not followed by a second
                        self.single_blink_counter += 1
                        event_detected = "Single Blink"
                        # Audio already played on first part, but if you want confirmation:
                        # enqueue_audio(BLINK_SOUNDS.get("Single Blink"))
                        
                    elif self.last_event is None:
                        # FIRST part of a blink/double blink sequence
                        self.single_blink_counter += 1
                        event_detected = "Single Blink"
                        enqueue_audio(BLINK_SOUNDS.get("Single Blink"))

                    self.last_event = event_detected
                    self.last_event_time = current_time_for_blink
                    
                self.blink_counter = 0

            # Display Blink Status
            blink_text = f"Blinks: {self.double_blink_counter*2 + self.single_blink_counter}" # Simple counter for total
            cv2.putText(img, blink_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return img

# =========================================================
# 5. STREAMLIT APP
# =========================================================
st.set_page_config(page_title="Unified Caregiver Instruction System", layout="wide")
st.title("🧠 Unified Real-time Detection System for Caregiver Instructions")
st.markdown("""
This system uses **Hand Gestures**, **Head Pose**, and **Blink Patterns** to trigger instructions.

---
**Instructions:**
1.  **Hand Gestures:** Show specific hand signs (e.g., one finger for "Bring Water").
2.  **Head Pose:** Hold your head in a direction (Up/Down/Left/Right) for **0.5 seconds**.
3.  **Blinks:** A simple blink registers as the first part of a potential double-blink command. A **Double Blink** (two quick blinks) will trigger a specific command.

**NOTE:** Audio will only work if the specified file paths are valid on the server/deployment environment.
""")

webrtc_streamer(
    key="unified_detection_system",
    video_transformer_factory=UnifiedVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    # Ensure Streamlit is running with a recent version for WebRTC support
)

# Clean up Pygame resources when Streamlit is shut down
def cleanup_audio():
    if pygame_inited:
        audio_queue.put(None) # Signal the worker to stop
        # Note: In a deployed environment, this might not always execute cleanly.

if st.button("Stop Stream and Audio"):
    # This button offers a manual way to stop the stream
    st.experimental_rerun()