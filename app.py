import streamlit as st

# =========================================================
# 0. STREAMLIT CONFIG (MUST BE THE FIRST STREAMLIT COMMAND)
# =========================================================
st.set_page_config(page_title="Unified Caregiver Instruction System", layout="wide")
st.title("🧠 Unified Real-time Detection System for Caregiver Instructions")
st.markdown("""
This system uses **Hand Gestures**, **Head Pose**, and **Blink Patterns** to trigger instructions for paralytic people.

---
**Detection Status:**
* **Hand Gestures:** Standard hand signs (e.g., index finger up).
* **Head Pose:** Stabilized detection after holding a direction (Up/Down/Left/Right) for **0.5 seconds**.
* **Blinks:** Tracks Single Blinks and commands are triggered by a **Double Blink** (two quick blinks).
""")

# =========================================================
# 1. IMPORTS & INITIAL SETUP
# =========================================================
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
# 2. AUDIO PATHS & SETUP
# =========================================================
# --- WARNING: REPLACE THESE LOCAL PATHS FOR DEPLOYMENT ---
# Example: r"C:\Users\nammi\...\bring water.mp3" should become "./Audios/bring water.mp3"

# Hand Gestures
HAND_GESTURE_SOUNDS = {
    "Bring Water": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\bring water.mp3",
    "Emergency": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\emergency.mp3",
    "Stop": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\stop.mp3",
    "Assist me outside": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\assist me outside.mp3",
    "Call 108": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\call 108.mp3",
    "Contact my caregiver": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\contact my caregiver.mp3",
    "Check my supplies": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\check my supplies.mp3",
    "Help me sit": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\help me sit.mp3"
}
# Head Poses
HEAD_POSE_SOUNDS = {
    "Up": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\bring_water.opus",
    "Down": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\emergency1.opus",
    "Right": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\hungry.opus",
    "Left": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\i_need_firstaid.opus"
}
# Blink Events
BLINK_SOUNDS = {
    "Single Blink": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\call_for_caretaker.opus",
    "Double Blink": r"C:\Users\nammi\OneDrive\Desktop\ML\Audios\need_help.opus"
}

pygame_inited = False
try:
    pygame.mixer.init()
    pygame_inited = True
except Exception as e:
    st.warning(f"Pygame mixer init failed: {e}. Audio disabled.")

audio_queue = Queue()
audio_lock = threading.Lock() if pygame_inited else None

def audio_worker():
    """Plays audio files from the queue sequentially."""
    while True:
        path = audio_queue.get()
        if path is None:
            audio_queue.task_done()
            break
        # Check if file exists, crucial for deployment
        if path and os.path.exists(path):
            try:
                if audio_lock:
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
    """Adds an audio path to the playback queue."""
    if pygame_inited and path:
        audio_queue.put(path)

# =========================================================
# 3. MEDIAPIPE & CONSTANTS SETUP
# =========================================================
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# Hand detection setup
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
# Face detection setup
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.6)

# Head Pose Constants
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), 
    (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0), (0.0, -330.0, -65.0) 
]) 
FACE_LANDMARK_POINTS = [1, 33, 263, 61, 291, 152] # Nose, Left Eye, Right Eye, Left Mouth, Right Mouth, Chin
HEAD_YAW_THRESHOLD = 10
HEAD_PITCH_THRESHOLD = 12
HEAD_STABLE_TIME = 0.5

# Blink Constants
LEFT_EYE_EAR_INDICES = [362, 380, 374, 263, 386, 385]
RIGHT_EYE_EAR_INDICES = [33, 159, 158, 133, 153, 145]
EAR_THRESHOLD = 0.3
CONSEC_FRAMES = 4
DOUBLE_BLINK_MAX_INTERVAL = 0.8 

# Hand Gesture Constants
FINGER_TIPS = [8, 12, 16, 20] 

# =========================================================
# 4. HELPER FUNCTIONS
# =========================================================

# --- Hand Gesture Helpers ---
def fingers_up(hand_landmarks):
    # This simplified logic assumes the user is using the camera-facing hand.
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

# --- Head Pose Helpers ---
def get_head_pose(landmarks, img_w, img_h):
    image_points = np.array([
        (landmarks[idx].x * img_w, landmarks[idx].y * img_h) for idx in FACE_LANDMARK_POINTS
    ], dtype="double")

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs
    )

    if not success:
        return None, None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, _ = eulerAngles.flatten()[:3]
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

# --- Blink Detection Helpers ---
def eye_aspect_ratio(eye_landmarks_indices, face_landmarks):
    def get_coords(idx):
        return np.array([face_landmarks[idx].x, face_landmarks[idx].y])

    A = np.linalg.norm(get_coords(eye_landmarks_indices[1]) - get_coords(eye_landmarks_indices[5]))
    B = np.linalg.norm(get_coords(eye_landmarks_indices[2]) - get_coords(eye_landmarks_indices[4]))
    C = np.linalg.norm(get_coords(eye_landmarks_indices[0]) - get_coords(eye_landmarks_indices[3]))
    
    return (A + B) / (2.0 * C) if C != 0 else 0.0

# =========================================================
# 5. VIDEO TRANSFORMER CLASS (Unified Logic)
# =========================================================
class UnifiedVideoTransformer(VideoTransformerBase):
    def __init__(self):
        # Head Pose State
        self.stable_direction = "Center"
        self.last_detected_direction = "Center"
        self.direction_start_time = time.time()
        
        # Blink Counting State
        self.blink_counter = 0 # frames below threshold
        self.double_blink_counter = 0 # total double blinks
        self.last_event = None # "Single Blink"
        self.last_event_time = 0.0
        
        # Hand Gesture State
        self.last_gesture = None
        self.last_head_direction = None

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
            
            # Draw Face Mesh
            mp_draw.draw_landmarks(img, face_result.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS)
            
            # 1. HEAD POSE LOGIC
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
                            # elif self.stable_direction == "Center": # Optionally stop audio here
                            #     pass 

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
                    # Blink occurred (end of closure)
                    current_time_for_blink = time.time()
                    
                    if self.last_event == "Single Blink" and (current_time_for_blink - self.last_event_time) < DOUBLE_BLINK_MAX_INTERVAL: 
                        # Detected a second blink within the time interval
                        self.double_blink_counter += 1
                        event_detected = "Double Blink"
                        enqueue_audio(BLINK_SOUNDS.get("Double Blink"))
                        self.last_event = None # Reset for next cycle
                    elif self.last_event is None:
                        # Detected a single blink, now waiting for a potential second
                        event_detected = "Single Blink"
                        enqueue_audio(BLINK_SOUNDS.get("Single Blink"))
                        self.last_event = event_detected
                        self.last_event_time = current_time_for_blink

                    self.blink_counter = 0
                elif self.last_event == "Single Blink" and (time.time() - self.last_event_time) >= DOUBLE_BLINK_MAX_INTERVAL:
                    # Single blink interval expired, reset without new event
                    self.last_event = None
                
                self.blink_counter = 0

            # Display Blink Status
            cv2.putText(img, f"Double Blinks: {self.double_blink_counter}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            if self.last_event:
                cv2.putText(img, f"Last: {self.last_event}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img

# =========================================================
# 6. STREAMLIT WEBRTC CALL
# =========================================================

webrtc_streamer(
    key="unified_detection_system",
    video_transformer_factory=UnifiedVideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)