import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load model
model_dict = pickle.load(open('model.pickle', 'rb'))
model = model_dict['model']

# create hand landmarks
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True,
                       model_complexity=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7,
                       max_num_hands=1)

# Create camera window
cap = cv2.VideoCapture(0)
try:
    while cap.isOpened():
        data_aux = []

        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        processed = hands.process(frameRGB)
        if processed.multi_hand_landmarks:
            for lm in processed.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            for lm in processed.multi_hand_landmarks:
                for i in range(len(lm.landmark)):
                    x = lm.landmark[i].x
                    y = lm.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            pred = model.predict([np.asarray(data_aux)])

            # pref_chr = labels[int(pred[0])]

            print(pred)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
