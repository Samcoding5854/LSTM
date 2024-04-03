from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import cv2


from ultralytics import YOLO
import numpy as np
import os
# Load a model

modelpose = YOLO('yolov8n-pose.pt') 

# Define path to the main folder containing class subfolders
data_path = "Dataset"  # Replace with your actual path

# Define actions based on subfolder names (modify as needed)
actions = np.array(os.listdir(data_path))  # Get list of subfolders
print(actions)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(120,34)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


model.load_weights('action.h5')


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


# 1. New detection variables
import cv2
sequence = []
sentence = []
threshold = 0.8

cap = cv2.VideoCapture("Dataset/Squat/squat_21.mp4")
# Set mediapipe model 

while cap.isOpened():

    # Read feed
    ret, frame = cap.read()
    image = frame

    # Make detections
    results = modelpose.predict(frame)
    # Flatten keypoints
    keypoints = np.array(results[0][0].keypoints.xyn).flatten()

    sequence.append(keypoints)
    sequence = sequence[-30:]
    
    if len(sequence) == 30:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        print(actions[np.argmax(res)])
        
        
    #3. Viz logic
        if res[np.argmax(res)] > threshold: 
            if len(sentence) > 0: 
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
            else:
                sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5: 
            sentence = sentence[-5:]

        # Viz probabilities
        image = prob_viz(res, actions, image, colors)
        
    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, ' '.join(sentence), (3,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Show to screen
    cv2.imshow('OpenCV Feed', image)

    # Break gracefully
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()