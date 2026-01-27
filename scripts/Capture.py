import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

csv_file = os.path.join('data', 'processed', 'hand_signs.csv')

# Check if CSV file exists, if not, create it with headers
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = ['label']
        for i in range(21):
            headers.extend([f'x{i}', f'y{i}', f'z{i}'])
        writer.writerow(headers)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw labels (optional, for visualization)
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.putText(frame, str(id), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif 97 <= key <= 122:  # a-z
        label = chr(key).upper()  # Convert to uppercase for label (e.g., 'A')
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
            hand_landmarks = results.multi_hand_landmarks[0]  # Take the first hand
            row = [label]
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"Saved data for label: {label}")
        else:
            print("No hand detected, skipping save.")

cap.release()
cv2.destroyAllWindows()