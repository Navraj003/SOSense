import cv2
import mediapipe as mp

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Function to detect SOS sign
def is_sos_gesture(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]

    fingers_folded = 0
    for i in range(1, 5):  # Skip thumb (index 0)
        tip_y = hand_landmarks.landmark[tip_ids[i]].y
        base_y = hand_landmarks.landmark[tip_ids[i] - 2].y
        if tip_y > base_y:  # finger is bent
            fingers_folded += 1

    # Thumb folded inside palm (basic check: thumb_tip x < thumb_ip x)
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]
    thumb_folded = thumb_tip.x < thumb_ip.x

    return fingers_folded == 4 and thumb_folded

# Webcam Feed
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            if is_sos_gesture(lm):
                cv2.putText(frame, " SOS SIGNAL DETECTED!", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("SOS Hand Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
