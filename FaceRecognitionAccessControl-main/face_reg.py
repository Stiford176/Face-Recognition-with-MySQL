import numpy as np
import os
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import face_recognition
from datetime import datetime
import dlib
from scipy.spatial import distance
import mysql.connector
import pickle

# Eye aspect ratio (EAR) calculation function for blink detection
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Indices for eye landmarks
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# Blink detection parameters
BLINK_THRESHOLD = 0.15  
BLINK_FRAMES = 0.50  
blink_counter = 0
blinks_detected = 0

# Load face detector and shape predictor
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

# MySQL database connection setup
def init_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",  
        password="123456",  
        database="face_recognition_db"
    )

# Function to load known faces from the database
def load_known_faces(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM faces")
    rows = cursor.fetchall()
    known_face_encodings = [pickle.loads(row[1]) for row in rows]
    known_face_names = [row[0] for row in rows]
    cursor.close()
    return known_face_encodings, known_face_names

# GUI setup
window = tk.Tk()
window.wm_title("Face Recognition for Access Control")
window.config(background="#000000")

imageFrame = tk.Frame(window, width=1920, height=1080, bg="black")
imageFrame.grid(row=2, column=0, columnspan=3, padx=1, pady=2) 

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Camera not detected! Exiting...")
    exit()

recognized_label = tk.Label(window, text="Recognized: None", font="Helvetica 16 bold", fg="black", bg="white")
recognized_label.grid(row=1, column=1, padx=10, pady=5)

welcome = tk.Label(window, text="", font="Helvetica 16 bold", fg="black", bg="white")
welcome.grid(row=0, column=1, padx=10, pady=5)

conn = init_db()
known_face_encodings, known_face_names = load_known_faces(conn)

face_names = []
is_registering = False  

def show_frame():
    global blink_counter, blinks_detected, face_names, is_registering  

    if is_registering:
        return  

    face_names = []  
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Error: No frame captured!")
        return

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    detected_real_face = False  

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Invalid or Not Registered"  

        if True in matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            name = known_face_names[best_match_index]
            detected_real_face = True  

        face_names.append(name)

        top, right, bottom, left = [int(i * 2) for i in (top, right, bottom, left)]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = [(p.x, p.y) for p in shape.parts()]
            left_eye = [shape[i] for i in LEFT_EYE]
            right_eye = [shape[i] for i in RIGHT_EYE]

            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            if avg_EAR < BLINK_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= BLINK_FRAMES:
                    blinks_detected += 1
                blink_counter = 0

    if detected_real_face:
        welcome.config(text=f"Hi {face_names[0]}! Welcome!", fg="black")
    else:
        welcome.config(text="⚠️ Only real humans can be recognized!", fg="red")

    recognized_label.config(text="Recognized: " + ", ".join(face_names) if face_names else "Recognized: Unknown")

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
    imgtk = ImageTk.PhotoImage(image=img)
    display1.configure(image=imgtk)
    display1.image = imgtk

    window.after(10, show_frame)

def register():
    global is_registering  

    if blinks_detected < 2:
        messagebox.showwarning("Warning", "Please blink twice to confirm liveliness.")
        return

    is_registering = True  

    log_dir = os.path.join(os.getcwd(), 'people')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    for name in face_names:
        if name == "Invalid or Not Registered":
            name = simpledialog.askstring("Register Face", "Please enter your name:")
            if name:
                ret, frame = cap.read()  
                if ret:
                    face_encoding = face_recognition.face_encodings(frame)[0]  
                    encoded_face = pickle.dumps(face_encoding)

                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO faces (name, encoding) VALUES (%s, %s) ON DUPLICATE KEY UPDATE encoding=%s",
                        (name, encoded_face, encoded_face)
                    )
                    conn.commit()
                    cursor.close()

                    log_file_path = os.path.join(log_dir, f"logs_{str(datetime.now())[:10]}.txt")
                    with open(log_file_path, 'a+') as f:
                        f.write(f"{str(datetime.now())[11:-10]}\t{name}\n")

                    welcome.config(text=f"Hi {name.capitalize()}! You are now registered.", fg="green")
                    messagebox.showinfo("Success", f"Face registered as {name}!")

    is_registering = False  

btn = tk.Button(window, text="Register", font="Helvetica 16 bold", fg="black", bg="white",
                command=register)
btn.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

display1 = tk.Label(imageFrame)
display1.grid(row=2, column=0, columnspan=3, padx=1, pady=2)

show_frame()

window.mainloop()
