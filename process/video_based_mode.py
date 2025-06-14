import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
from collections import Counter


label = "Warmup...."
predictions = np.array([])

event_label_updated = threading.Event()
event_predictions_updated = threading.Event()


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("./model.h5")

# cap = cv2.VideoCapture("deadlift_1.mp4")

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_world_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        if lm.visibility < 0.55 and id > 0:
            prev_lm = results.pose_landmarks.landmark[id - 1]
            next_lm = results.pose_landmarks.landmark[id + 1] if id + 1 < len(results.pose_landmarks.landmark) else prev_lm

            cx = int((prev_lm.x + next_lm.x) / 2 * w)
            cy = int((prev_lm.y + next_lm.y) / 2 * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    if (label == "NOT DETECTED"):
        fontColor = (0, 0, 255)
    elif (label == "Warmup...."):
        fontColor = (0, 165, 255)
    else:
        fontColor = (0, 255, 0)
    thickness = 2
    lineType = cv2.LINE_AA
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list):
    global predictions
    event_predictions_updated.clear() # Clear previous predictions

    # Prepare the input data for prediction
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0) # Add batch dimension for LSTM input

    # Make prediction using LSTM model
    predictions = model.predict(lm_list) 
    print(predictions)

    # Set the print options for displaying predictions
    np.set_printoptions(suppress=True, precision=6)
    event_predictions_updated.set() # Notify that predictions have been updated

    # Map predictions to labels
    detect_label(predictions)



def detect_label(predictions):
    global label
    event_label_updated.clear() # Clear previous labels

    # Define the prediction threshold
    threshold = 0.9
    
    # Map each index to a corresponding exercise label
    label_map = {
        0: "BARBELL BICEPS CURL",
        1: "BENCH PRESS",
        2: "CHEST FLY MACHINE",
        3: "DEADLIFT",
        4: "DECLINE BENCH PRESS",
        5: "HAMMER CURL",
        6: "HIP THRUST",
        7: "INCLINE BENCH PRESS",
        8: "LAT PULLDOWN",
        9: "LATERAL RAISE",
        10: "LEG EXTENSION",
        11: "LEG RAISES",
        12: "PLANK",
        13: "PULL UP",
        14: "PUSH UP",
        15: "ROMANIAN DEADLIFT",
        16: "RUSSIAN TWIST",
        17: "SHOULDER PRESS",
        18: "SQUAT",
        19: "T BAR ROW",
        20: "TRICEP DIPS",
        21: "TRICEP PUSHDOWN"
    }
    
    # Iterate through the predictions and map to the appropriate label
    for i, prediction in enumerate(predictions[0]):
        if prediction >= threshold:
            label = label_map.get(i, "NOT DETECTED")  # If no match, set "NOT DETECTED"
            break

    # Notify that the label has been updated
    event_label_updated.set()
    return label
    

# Count the number of occurrences of each element
def most_common_element(lst):
    count = Counter(lst)  
    return count.most_common(1)[0][0] 

# def main(): 
#     # average_accuray = None
#     global label
#     global predictions
#     sum_predictions =0
#     i = 0
#     warmup_frames = 15

#     # Define variables for number of time steps (frames) and initialize list to store keypoints
#     n_time_steps = 35
#     lm_list = []
#     time_predictions = 0

    
#     list_label = []
#     while True:

#         ret, img = cap.read()
#         if not ret:
#             break
#         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = pose.process(imgRGB)
#         i = i + 1
    
#         if i > warmup_frames:

#             # Check if pose landmarks are available
#             if results.pose_world_landmarks:
#                 c_lm = make_landmark_timestep(results) # Extract 3D keypoints
#                 lm_list.append(c_lm) # Add keypoints for this frame to the list

#                 if len(lm_list) == n_time_steps: # When enough 35 frames are collected, pass them to LSTM for prediction
#                     # Using threading to predict asynchronously
#                     t1 = threading.Thread(target=detect, args=(model, lm_list,))
#                     t1.start()

#                     list_label.append(label)
#                     if predictions.size > 0:
#                         sum_predictions += predictions[0][:]
#                         time_predictions+=1
#                     lm_list = []

#                 img = draw_landmark_on_image(mpDraw, results, img)
#             else: 
#                 label = "Waiting..."
#         img = draw_class_on_image(label, img)
#         cv2.imshow("Image", img)
#         if cv2.waitKey(1) == ord('q'):
#             break
#         if i == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
#             average_predictions = sum_predictions/time_predictions
#             average_predictions = np.expand_dims(average_predictions, axis=0)
#             main_label = detect_label(average_predictions)
#             print(main_label)
    

#     label_most = most_common_element(list_label)
#     print(label_most)
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()