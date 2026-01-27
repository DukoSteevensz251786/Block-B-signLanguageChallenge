import cv2
import mediapipe as mp
import pickle
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,               # usually one hand for alphabet
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Load the model
with open(r'models/sign_model_rf.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    predicted_label = "No hand"

    if results.multi_hand_landmarks:
        # Take the first (usually dominant) hand
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) #DEBUGGING, SHOW CONNECTIONS
        
        # Prepare features (same order as training)
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        X = np.array(landmarks).reshape(1, -1)  # shape (1, 63)

        # Predict
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0].max()

        if prob > 0.7:  # confidence threshold — tune as needed
            predicted_label = f'{pred}'
            #predicted_label = f"{pred} ({prob:.2f})"
        elif prob > 0.3 and prob < 0.7:
            predicted_label = f"? ({pred}) ({prob:.2f})"
        else:
            predicted_label = "Not recognised"
        

    # Show prediction
    cv2.putText(frame, predicted_label, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20,188,255), 3)
    

    cv2.imshow("Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()