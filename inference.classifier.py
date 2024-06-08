import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the model
model_dict = pickle.load(open('./modelsibi.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.5)

# Labels dictionary mapping indices to letters (A-Z)
labels_dict = {i: chr(i + ord('A')) for i in range(26)}

while True:
    data_aux = []
    x_ = []
    y_ = []
    z_ = []  # List for z coordinates
    ret, frame = cap.read()

    if not ret:
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                z = landmark.z  # Add the z coordinate
                data_aux.append(x)
                data_aux.append(y)
                data_aux.append(z)  # Include z coordinate in data_aux
                x_.append(x)
                y_.append(y)
                z_.append(z)  # Include z coordinate in respective list

        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        # Ensure the feature vector length is correct
        if len(data_aux) == 63:  # Check if the number of features is correct
            # Append zeros to match the 84 features expectation
            data_aux.extend([0] * (84 - len(data_aux)))
            print("Data aux:", data_aux)
            prediction = model.predict([np.asarray(data_aux)])
            print("Prediction:", prediction)
            try:
                prediction_value = int(prediction[0])
                predicted_character = labels_dict[prediction_value]
            except (ValueError, KeyError):
                # Handle the case where the value cannot be converted to an integer
                # or the predicted character is not in the labels dictionary
                print("Error: Could not determine predicted character.")
                predicted_character = '?'  # Placeholder for unknown character

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, predicted_character, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)  # Shadow effect

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
