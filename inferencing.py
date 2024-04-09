from ultralytics import YOLO
import numpy as np
import dv_processing as dv  # Import the dv_processing library for DAVIS 346 access

cameras = dv.io.discoverDevices()

print(f"Device discovery: found {len(cameras)} devices.")
for camera_name in cameras:
    print(f"Detected device [{camera_name}]")

# Open any camera
capture = dv.io.CameraCapture()

# Print the camera name
print(f"Opened [{capture.getCameraName()}] camera, it provides:")


# Check whether event stream is available
if capture.isEventStreamAvailable():
    # Get the event stream resolution
    resolution = capture.getEventResolution()

    # Print the event stream capability with resolution value
    print(f"* Events at ({resolution.width}x{resolution.height}) resolution")

# Check whether frame stream is available
if capture.isFrameStreamAvailable():
    # Get the frame stream resolution
    resolution = capture.getFrameResolution()

    # Print the frame stream capability with resolution value
    print(f"* Frames at ({resolution.width}x{resolution.height}) resolution")

# Check whether the IMU stream is available
if capture.isImuStreamAvailable():
    # Print the imu data stream capability
    print("* IMU measurements")

# Check whether the trigger stream is available
if capture.isTriggerStreamAvailable():
    # Print the trigger stream capability
    print("* Triggers")
    
    
# Load a model
modelpose = YOLO('yolov8n-pose.pt') 

  
import os

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Thirty videos worth of data
no_sequences = 46

# Videos are going to be 30 frames in length
sequence_length = 15



# Define path to the main folder containing class subfolders
data_path = "C:\\Users\\ag701\\Desktop\\lstm\\LSTM\\MP_Data"  # Replace with your actual path

# Define actions based on subfolder names (modify as needed)
actions = np.array(os.listdir(data_path))  # Get list of subfolders
print(actions)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)



model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(15,34)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.load_weights('C:\\Users\\ag701\\Desktop\\lstm\\LSTM\\action15Frames96.h5')


import cv2 

colors = [(245,117,16), (117,245,16), (16,117,245), (126, 249, 255), (255, 166, 255),(16,117,245), (40, 166, 133)]
def prob_viz(res, actions, input_frame, colors, keypoints):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    top_left_x = int(keypoints[0][:, 0].min().item())  # Minimum x-coordinate across all keypoints
    top_left_y = int(keypoints[0][:, 1].min().item())-20  # Minimum y-coordinate across all keypoints
    bottom_right_x = int(keypoints[0][:, 0].max().item())  # Maximum x-coordinate across all keypoints
    bottom_right_y = int(keypoints[0][:, 1].max().item())+20 # Maximum y-coordinate across all keypoints

    # Define bounding box thickness and color
    thickness = 2
    color = (0, 255, 0)  # Green for bounding box

    # Draw the rectangle
    cv2.rectangle(output_frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness)

    return output_frame





# 1. New detection variables
import cv2
import torch

sequence = []
sentence = []
threshold = 0.6
save_directory = "CapturedFrames"
os.makedirs(save_directory, exist_ok=True)

# cap = cv2.VideoCapture("C:\\Users\\ag701\\Desktop\\lstm\\LSTM\\jumping1.mp4")


# Use dv_processing for DAVIS 346 access
cap = dv.io.CameraCapture()


# Initiate a preview window
cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
frame_counter = 0

while cap.isRunning():
    # Read feed
    frame = cap.getNextFrame()

    # if not ret:
    #     print("No frames left in video. Exiting...")
    #     break
    
    if frame is None:
        print("No frame received from DAVIS 346 camera.")
    
    else:
        
        image = frame

        filename = f"{save_directory}/frame_{frame_counter}.jpg"
        cv2.imwrite(filename, frame.image)
        frame_counter += 1

        # Make detections
        results = modelpose.predict(filename)
        
        keyframes = []
        # # Flatten keypoints
        keypointsn = np.array(results[0][0].keypoints.xyn).flatten()
        
        if len(keypointsn) == 0:
                keypointsn = torch.zeros(1, 17, 2)
            
        for r in results:
            keypoints=[]
            keypoints = r.keypoints.xy
            print("keypoints", keypoints.shape)
            # if len(keypoints) > 0:  # Check for missing keypoints (optional)
            #     keyframes.append(keypoints)
            if len(keypoints) == 0:
                keypoints = torch.zeros(1, 17, 2)
        # if len(keyframes) > 0:  # Check for empty frames (optional)
        #     keyframes = np.array(keyframes).flatten()
        #     sequence.append(keyframes[-1])
        #     sequence = sequence[-15:]

        sequence.append(keypointsn)
        sequence = sequence[-15:]
    
        if len(sequence) == 15:
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
            image = prob_viz(res, actions, image, colors, keypoints)
        
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