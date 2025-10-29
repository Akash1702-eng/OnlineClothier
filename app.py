from flask import Flask, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import math

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

@app.route('/')
def home():
    return "Online Clothier ML Model is running!"

@app.route('/get_measurements', methods=['GET'])
def get_measurements():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Camera not found"}), 500

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 500

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if not results.pose_landmarks:
        return jsonify({"error": "Pose not detected"}), 400

    h, w, _ = frame.shape
    lm = results.pose_landmarks.landmark

    # Calculate approximate measurements (example logic)
    height_pixels = math.dist(
        (lm[mp_pose.PoseLandmark.NOSE].x * w, lm[mp_pose.PoseLandmark.NOSE].y * h),
        (lm[mp_pose.PoseLandmark.LEFT_ANKLE].x * w, lm[mp_pose.PoseLandmark.LEFT_ANKLE].y * h)
    )

    # Approx conversion to inches (just an example scaling)
    height_inches = round(height_pixels / 10, 2)
    chest_size = round(height_inches * 0.35, 2)
    shoulder_size = round(height_inches * 0.25, 2)
    bottom_length = round(height_inches * 0.5, 2)

    return jsonify({
        "height": height_inches,
        "chest_size": chest_size,
        "shoulder_size": shoulder_size,
        "bottom_length": bottom_length
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
