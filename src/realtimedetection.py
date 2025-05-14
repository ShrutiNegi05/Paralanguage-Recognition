import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load Models
isl_model = load_model("C:\\Users\\Dell\\OneDrive\\Documents\\merge_emotions_signs\\isl_model.h5")
emotion_model = load_model("C:\\Users\\Dell\\OneDrive\\Documents\\merge_emotions_signs\\retrained_emotion_model.h5")

# Labels
isl_labels = ['hello', 'goodbye', 'namaste', 'i love you', 'yes', 'no', 'sorry', 'thank you']
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprised']

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Text-to-Speech setup
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Logging setup
log_file = open("interpreter_log.txt", "a")

# State memory for cooldown
prev_sign = ""
prev_emotion = ""
speak_cooldown = 30  # Frames to wait before speaking again
cooldown_counter = 0

# Start Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ================= HAND + ISL PREDICTION =================
    result = hands.process(frame_rgb)
    isl_prediction = "No Sign"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]

            xmin = int(min(x_list) * w) - 20
            xmax = int(max(x_list) * w) + 20
            ymin = int(min(y_list) * h) - 20
            ymax = int(max(y_list) * h) + 20

            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

            hand_img = frame[ymin:ymax, xmin:xmax]

            try:
                hand_img = cv2.resize(hand_img, (224, 224))
                hand_img = hand_img.astype("float32") / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)

                isl_pred = isl_model.predict(hand_img, verbose=0)
                isl_prediction = isl_labels[np.argmax(isl_pred)]

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            except:
                isl_prediction = "Hand crop error"

    # ================= EMOTION PREDICTION =================
    emotion_prediction = "No Face"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w_f, h_f) in faces:
        face_roi = frame[y:y+h_f, x:x+w_f]

        face_roi = cv2.resize(face_roi, (128, 128))
        face_roi = face_roi.astype("float") / 255.0
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)

        emotion_pred = emotion_model.predict(face_roi, verbose=0)
        emotion_prediction = emotion_labels[np.argmax(emotion_pred)]

        cv2.rectangle(frame, (x, y), (x+w_f, y+h_f), (0, 255, 0), 2)
        break

    # ================= DISPLAY + SPEECH =================
    cv2.putText(frame, f"Sign: {isl_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
    cv2.putText(frame, f"Emotion: {emotion_prediction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)

    if isl_prediction != "No Sign" and emotion_prediction != "No Face":
        sentence = f"You look {emotion_prediction} and you signed {isl_prediction}"
        cv2.putText(frame, sentence, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if (isl_prediction != prev_sign or emotion_prediction != prev_emotion) and cooldown_counter <= 0:
            tts_engine.say(sentence)
            tts_engine.runAndWait()

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"{timestamp}, Emotion: {emotion_prediction}, Sign: {isl_prediction}\n")
            log_file.flush()

            prev_sign = isl_prediction
            prev_emotion = emotion_prediction
            cooldown_counter = speak_cooldown
        else:
            cooldown_counter -= 1

    cv2.imshow("ISL + Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
