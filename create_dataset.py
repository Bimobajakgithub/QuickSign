import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)  # Increased confidence

DATA_DIR = './SIBI'

data = []
labels = []

# Check if the DATA_DIR exists
if not os.path.exists(DATA_DIR):
    print(f"Data directory {DATA_DIR} does not exist.")
else:
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):
            continue
        
        for img_path in os.listdir(dir_path):
            img_file_path = os.path.join(dir_path, img_path)
            print(f"Processing image: {img_file_path}")
            
            data_aux = []
            img = cv2.imread(img_file_path)
            if img is None:
                print(f"Failed to read image: {img_file_path}")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)

                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"No hands detected in image: {img_file_path}")

# Save the dataset if data is collected
if data and labels:
    with open('datasibi.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"Dataset created with {len(data)} samples.")
else:
    print("No data collected. Please check the dataset creation process.")
