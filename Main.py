import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to main images folder (each subfolder = person name)
path = 'Images'
encode_list_Known = []
PersonName = []


# Loop through each folder (person)
for person_folder in os.listdir(path):
    person_path = os.path.join(path, person_folder)
    if not os.path.isdir(person_path):
        continue

    encodings = []
    for image_file in os.listdir(person_path):
        img_path = os.path.join(person_path, image_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_enc = face_recognition.face_encodings(img_rgb)
        if face_enc:
            encodings.append(face_enc[0])

    if encodings:
        avg_encoding = np.mean(encodings, axis=0)
        encode_list_Known.append(avg_encoding)
        PersonName.append(person_folder)
        print(f"✅ Trained: {person_folder} ({len(encodings)} images)")

print("🎯 Training complete! Total people:", len(PersonName))

# Attendance function
def attendance(name):
    with open('Attendance.csv', 'a') as f:
        time_now = datetime.now()
        tStr = time_now.strftime('%H:%M:%S')
        dStr = time_now.strftime('%d/%m/%Y')
        f.write(f"{name},{tStr},{dStr}\n")

cap = cv2.VideoCapture(0)
process_this_frame = True  # Frame skip toggle

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # HOG model for faster detection
        faces_currentframe = face_recognition.face_locations(rgb_small_frame, model="hog")
        encode_currentframe = face_recognition.face_encodings(rgb_small_frame, faces_currentframe)

        names = []
        colors = []
        for encodeFace in encode_currentframe:
            faceDistance = face_recognition.face_distance(encode_list_Known, encodeFace)
            best_match_index = np.argmin(faceDistance) if len(faceDistance) > 0 else -1

            if best_match_index != -1 and faceDistance[best_match_index] < 0.5:
                name = PersonName[best_match_index].upper()
                color = (0, 255, 0)
                attendance(name)
            else:
                name = "UNKNOWN"
                color = (0, 0, 255)

            names.append(name)
            colors.append(color)

    process_this_frame = not process_this_frame  # Skip alternate frames

    # Draw results
    if 'faces_currentframe' in locals():
        for (faceLoc, name, color) in zip(faces_currentframe, names, colors):
            y1, x2, y2, x1 = [v * 4 for v in faceLoc]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y2 - 25), (x2, y2), color, cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Title bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (50, 50, 50), -1)
    cv2.putText(frame, "Face Recognition Attendance System", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Face Recognition Attendance", frame)
    if cv2.waitKey(1) & 0xFF == 13:  # Enter key
        break

cap.release()
cv2.destroyAllWindows()
