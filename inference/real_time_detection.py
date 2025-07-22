import os
import smtplib
import sys
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import cv2
import joblib
import mediapipe as mp
import numpy as np
from keras.models import load_model
from twilio.rest import Client

# Project structure adjustment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.feature_extractor import extract_features

# --- CONFIG ---
gesture_labels = {
    0: "sfh",
    1: "need_help",
    2: "need_police",
    3: "need_ambulance",
    4: "visit_me",
    5: "call_me",
    6: "not_yet_helped",
    7: "i_am_okay",
    8: "neutral"
}

CONFIDENCE_THRESHOLD = 0.7
ALERT_COOLDOWN_SECONDS = 60
WARMUP_SECONDS = 2

# Email config

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# --- Load Models ---
model = load_model('models/gesture_classifier.h5')
scaler = joblib.load("models/scaler.pkl")

# --- Email ---
def send_email_alert():
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = "🚨 SOS Detected from SOSense App"
    msg.attach(MIMEText("An SOS gesture has been detected. Please check on the user immediately.", 'plain'))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print("[INFO] Email sent successfully.")
    except Exception as e:
        print("[ERROR] Failed to send email:", e)

# --- SMS ---
def send_sms_alert():
    try:
        message = client.messages.create(
            body="🚨 SOS Detected. Immediate action required!",
            from_=TWILIO_FROM_NUMBER,
            to=TO_NUMBER
        )
        print("[INFO] SMS sent:", message.sid)
    except Exception as e:
        print("[ERROR] SMS failed:", e)

# --- MediaPipe ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- Webcam & Loop ---
cap = cv2.VideoCapture(0)
last_sent = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if time.time() - start_time < WARMUP_SECONDS:
        cv2.putText(frame, "Warming up...", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("SOSense - Real-time Gesture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            landmarks = [(pt.x, pt.y, pt.z) for pt in lm.landmark]
            features = extract_features(landmarks)
            scaled_features = scaler.transform([features])
            input_data = np.array(scaled_features)

            prediction = model.predict(input_data, verbose=0)
            predicted_label = np.argmax(prediction)
            confidence = np.max(prediction)

            gesture_name = gesture_labels.get(predicted_label, "Unknown")
            print(f"[DEBUG] Prediction: {prediction}")
            print(f"[DEBUG] Predicted index: {predicted_label}, Gesture: {gesture_name}, Confidence: {confidence:.2f}")

            cv2.putText(frame, f"Detected: {gesture_name} ({confidence:.2f})", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # SOS Detection
            if predicted_label == 0 and confidence > CONFIDENCE_THRESHOLD:
                cv2.putText(frame, "!!! SOS SIGNAL DETECTED !!!", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                current_time = time.time()
                if current_time - last_sent > ALERT_COOLDOWN_SECONDS:
                    print("[INFO] Triggering ALERTS...")
                    send_email_alert()
                    send_sms_alert()
                    last_sent = current_time

    cv2.imshow("SOSense - Real-time Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
