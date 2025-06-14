from flask import Flask, request, render_template, send_from_directory, session, Response
import os
import tensorflow as tf
import cv2
import numpy as np
import process.video_based_mode as video_based_mode
from process.video_based_mode import make_landmark_timestep, draw_landmark_on_image, draw_class_on_image, detect,detect_label, pose, mpDraw, event_label_updated, event_predictions_updated, most_common_element
from process.real_time_mode import real_time_detection
import threading
from flask_socketio import SocketIO         # type: ignore
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
POSE_FOLDER = "static/poses"

app.secret_key = "supersecretkey"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["POSE_FOLDER"] = POSE_FOLDER

socketio = SocketIO(app)

label_most = None

model = tf.keras.models.load_model("model.h5")

# Create the folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(POSE_FOLDER):
    os.makedirs(POSE_FOLDER)



def process_video(video_path, output_path):
    global label_most
    i = 0
    sum_predictions =0
    time_predictions = 0
    warmup_frames = 15
    n_time_steps = 35
    lm_list = []
    list_label = []
    # global label
    cap = cv2.VideoCapture(video_path)

    # Get video information
    fourcc = cv2.VideoWriter_fourcc(*"AVC1")  # Codec to write mp4 files
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        i = i + 1
        if i > warmup_frames:
            if results.pose_world_landmarks:
                c_lm = make_landmark_timestep(results)

                lm_list.append(c_lm)
                if len(lm_list) == n_time_steps:
                    event_label_updated.clear()  # Reset the event before making a prediction
                    event_predictions_updated.clear()
                    t1 = threading.Thread(target=detect, args=(model, lm_list))
                    t1.start()
                    event_predictions_updated.wait()
                    event_label_updated.wait()  # Wait until `detect()` updates the mode label
                    list_label.append(video_based_mode.label)
                    if video_based_mode.predictions.size > 0:
                        sum_predictions += video_based_mode.predictions[0][:]
                        time_predictions+=1
                    lm_list = []

                img = draw_landmark_on_image(mpDraw, results, img)
            else: 
                label = "Waiting..."
        img = draw_class_on_image(video_based_mode.label, img)

        if i == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            average_predictions = sum_predictions/time_predictions
            average_predictions = np.expand_dims(average_predictions, axis=0)
            main_label = detect_label(average_predictions)

        out.write(img)  # Write the frame to the video output
    label_most = most_common_element(list_label)
    cap.release()
    out.release()


@app.route("/", methods=["GET", "POST"])
def live_detection():
    # Return the HTML page with video stream embedded in a small frame
    return render_template("live_detection.html")

@app.route('/video_feed')
def video_feed():
    # Stream the live video feed for real-time detection in MJPEG format
    return Response(real_time_detection(model),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/videoclassification", methods=["GET", "POST"])
def video_classification():
    input_path = session.get("input_path")
    file_name = session.get("file_name")
    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename.endswith(".mp4"):
            input_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            input_path = input_path.replace("\\", "/")
            
            session["input_path"] = input_path
            session["file_name"] = file.filename
            file.save(input_path)
    return render_template("video_classification.html", video_url=input_path)

@app.route("/identify", methods=["GET", "POST"])
def identify():
    input_path = session.get("input_path")  
    file_name = session.get("file_name")
    output_path = os.path.join(app.config["POSE_FOLDER"], file_name)
    output_path = output_path.replace("\\", "/")
    process_video(input_path, output_path)
    # Render the results of the pose detection with the processed video and most common label
    return render_template("identify.html", video_url=output_path, label_most = label_most)

if __name__ == '__main__':
    socketio.run(app, debug=True)
