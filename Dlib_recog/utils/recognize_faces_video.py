import cv2
import dlib
import face_recognition
import imutils
import argparse
import pickle
import time

# Argument parser for command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str, help="path to the output video")
ap.add_argument("-y", "--display", type=int, default=1, help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection_method", type=str, default="cnn", help="face detection model to use: cnn or hog")
args = vars(ap.parse_args())

# Load the known faces and encodings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

print("[INFO] starting video stream...")
# stream_url = 'rtsp://admin:Vnpt@123@123.25.190.36:552/ch1/main/av_stream'   # Replace with your RTSP stream URL
# video_capture = cv2.VideoCapture(stream_url)
video_capture = cv2.VideoCapture(0)
# Initialize variables
trackers = []
face_names = []
fps_counter = 0
fps_start_time = time.time()
fps = 0  # Initialize FPS counter

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame to increase processing speed
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(rgb, width=750)
    r = frame.shape[1] / float(rgb.shape[1])

    # Update trackers and get names for recognized faces
    for tracker, name in zip(trackers, face_names):
        tracker.update(rgb)
        pos = tracker.get_position()
        cv2.rectangle(frame, (int(pos.left() * r), int(pos.top() * r)), (int(pos.right() * r), int(pos.bottom() * r)), (0, 255, 0), 2)
        cv2.putText(frame, name, (int(pos.left() * r), int(pos.top() * r) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Check to run detection and reset trackers
    if len(trackers) == 0:
        boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
        encodings = face_recognition.face_encodings(rgb, boxes)

        face_names = []
        trackers = []

        for encoding, box in zip(encodings, boxes):
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(box[3], box[0], box[1], box[2])
            tracker.start_track(rgb, rect)
            trackers.append(tracker)

            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matched_idxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                name = max(counts, key=counts.get)

            face_names.append(name)

    # Calculate FPS
    fps_counter += 1
    if (time.time() - fps_start_time) > 1:
        fps = fps_counter / (time.time() - fps_start_time)
        fps_counter = 0
        fps_start_time = time.time()

    # Display FPS on the frame
    cv2.putText(frame, f"FPS: {fps:.2f}", (1400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # Display the output frame
    if args["display"] > 0:
        cv2.imshow("DETECT", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Clean up
video_capture.release()
cv2.destroyAllWindows()
if args["output"] is not None and writer is not None:
    iter.release()
