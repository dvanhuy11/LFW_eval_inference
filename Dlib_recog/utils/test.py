import face_recognition
import argparse
import pickle
import cv2
import time 
# Khởi tạo các đối số đầu vào
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to the serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str, help="path to the output video")
ap.add_argument("-y", "--display", type=int, default=1, help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection_method", type=str, default="hog", help="face detection model to use: cnn or hog")
args = vars(ap.parse_args())

# Tải dữ liệu encoding khuôn mặt đã biết
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# Khởi tạo camera
camera = cv2.VideoCapture(0)
ret, frame = camera.read()
if not ret:
    print("[ERROR] Could not read frame from camera.")
    exit()

# Thiết lập codec và tạo đối tượng VideoWriter nếu cần lưu video
if args["output"] is not None:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args["output"], fourcc, 20.0, (frame.shape[1], frame.shape[0]))

while True:
    ret, frame = camera.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    start_time = time.time()
    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)

        names.append(name)

    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

    if args["output"] is not None:
        out.write(frame)

    end_time = time.time()
    fps_text = f"FPS: {1.0 / (end_time - start_time):.2f}"
    cv2.putText(frame, fps_text, (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)
    
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
if args["output"] is not None:
    out.release()
cv2.destroyAllWindows()
