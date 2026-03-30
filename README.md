🧠 Motion-Based Message Conveyor for Blind & Non-Verbal People
A real-time, audio-free assistive communication system that enables blind, paralyzed, or non-verbal individuals to convey caregiving instructions using Hand Gestures, Head Pose, and Eye Blink Patterns — powered entirely by computer vision.


📌 Overview
Millions of people with physical or neurological conditions are unable to speak or use traditional communication devices. This project provides a non-verbal, touchless communication channel using only a webcam — no special hardware, no audio required.
By tracking three simultaneous input modalities in real-time, patients can silently communicate critical needs like water, emergency help, food, and first aid to caregivers.

✨ Key Features

🖐️ Hand Gesture Recognition — 8 mapped gestures for common caregiver instructions
👀 Head Pose Estimation — Directional head movement (Up/Down/Left/Right) mapped to commands
👁️ Eye Blink Detection — Double blink triggers a command confirmation signal
📷 Real-time Webcam Stream — Runs directly in browser via Streamlit + WebRTC
🔇 Completely Audio-Free — Designed for silent, non-invasive use in care environments
⚡ Stabilized Detection — Head pose confirmed only after holding direction for 0.5 seconds to reduce false triggers


🗂️ Communication Mappings
🖐️ Hand Gestures
Finger PatternInstructionIndex finger upBring WaterIndex + Middle upEmergencyAll fingers downStopIndex + Middle + Ring upAssist me outsideAll fingers upCall 108Middle + Ring + Pinky upContact my caregiverPinky only upCheck my suppliesRing + Pinky upHelp me sit
👀 Head Pose
DirectionInstructionUpWATERDownEMERGENCYLeftFIRST AIDRightFOOD
👁️ Eye Blinks
PatternActionDouble BlinkTrigger COMMAND confirmation

🏗️ System Architecture
Webcam Input (WebRTC)
        ↓
   MediaPipe Processing
   ┌─────────────────────────────┐
   │  Hand Landmark Detection    │ → Gesture → Instruction
   │  Face Mesh (478 landmarks)  │ → Head Pose → Direction → Instruction
   │  Eye Landmark Tracking      │ → EAR → Blink → Command
   └─────────────────────────────┘
        ↓
   OpenCV Overlay (Real-time annotations)
        ↓
   Streamlit WebRTC Live Stream

🛠️ Tech Stack
ToolPurposePython 3.8+Core languageMediaPipeHand, face mesh, landmark detectionOpenCVFrame processing & annotationStreamlitWeb UIstreamlit-webrtcLive webcam streaming in browserNumPyMath & landmark coordinate operations

🚀 How to Run
bash# 1. Clone the repo
git clone https://github.com/YourUsername/motion-message-conveyor.git
cd motion-message-conveyor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser and allow webcam access.

---

## 📦 requirements.txt
```
streamlit
streamlit-webrtc
mediapipe
opencv-python
numpy
```

---

## 📁 Project Structure
```
motion-message-conveyor/
│
├── app.py               # Main Streamlit application
├── requirements.txt     # Dependencies
└── README.md            # Project documentation

💡 Use Cases

🏥 Hospital patients who are paralyzed or post-surgery
👴 Elderly individuals with speech impairments
🧏 Non-verbal individuals needing silent communication
🦮 Visually impaired caregiving scenarios where touch isn't possible


🔮 Future Improvements

 Text-to-speech output for detected commands
 Custom gesture training for personalized mappings
 Mobile-friendly version
 Multi-language instruction support
 Alert notification system for caregivers
