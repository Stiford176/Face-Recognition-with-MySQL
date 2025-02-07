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

# -----------------------------------------
#  🔹 Eye Aspect Ratio (EAR) for Blink Detection
# -----------------------------------------

def eye_aspect_ratio(eye):
    """
    Calculates the Eye Aspect Ratio (EAR) to detect blinks.
    EAR Formula: (vertical_dist1 + vertical_dist2) / (2 * horizontal_dist)
    """
    A = distance.euclidean(eye[1], eye[5])  # Vertical distance
    B = distance.euclidean(eye[2], eye[4])  # Vertical distance
    C = distance.euclidean(eye[0], eye[3])  # Horizontal distance
    return (A + B) / (2.0 * C)

# Indices for left and right eye landmarks (from Dlib's 68-face landmarks model)
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# Blink detection parameters
BLINK_THRESHOLD = 0.15  # EAR value below which a blink is detected
BLINK_FRAMES = 0.50  # Number of frames where EAR is below threshold to confirm blink
blink_counter = 0  # Counts consecutive frames with blink
blinks_detected = 0  # Total blinks detected

# -----------------------------------------
#  🔹 Dlib Face Landmark Detector
# -----------------------------------------

# Load the face detector and shape predictor for facial landmarks
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)  # Loads landmark model
detector = dlib.get_frontal_face_detector()  # Face detector

# -----------------------------------------
#  🔹 MySQL Database for Face Storage
# -----------------------------------------

def init_db():
    """
    Initializes the database connection to store face encodings.
    """
    return mysql.connector.connect(
        host="localhost",
        user="root",  
        password="123456",  
        database="face_recognition_db"
    )

def load_known_faces(conn):
    """
    Loads stored face encodings from the database.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name, encoding FROM faces")
    rows = cursor.fetchall()
    known_face_encodings = [pickle.loads(row[1]) for row in rows]  # Decode encodings
    known_face_names = [row[0] for row in rows]  # Store corresponding names
    cursor.close()
    return known_face_encodings, known_face_names

# -----------------------------------------
#  🔹 Tkinter GUI Setup
# -----------------------------------------

window = tk.Tk()
window.wm_title("Face Recognition for Access Control")
window.config(background="#000000")

# Frame for displaying camera feed
imageFrame = tk.Frame(window, width=1920, height=1080, bg="black")
imageFrame.grid(row=2, column=0, columnspan=3, padx=1, pady=2)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Camera not detected! Exiting...")
    exit()

# Labels to show recognition status
recognized_label = tk.Label(window, text="Recognized: None", font="Helvetica 16 bold", fg="black", bg="white")
recognized_label.grid(row=1, column=1, padx=10, pady=5)

welcome = tk.Label(window, text="", font="Helvetica 16 bold", fg="black", bg="white")
welcome.grid(row=0, column=1, padx=10, pady=5)

# Load stored face data from the database
conn = init_db()
known_face_encodings, known_face_names = load_known_faces(conn)

face_names = []
is_registering = False  # Flag to prevent multiple registrations at once

# -----------------------------------------
#  🔹 Face Recognition & Blink Detection
# -----------------------------------------

def show_frame():
    """
    Captures a frame, detects faces, checks for blinks, and displays the result.
    """
    global blink_counter, blinks_detected, face_names, is_registering  

    if is_registering:
        return  # Prevents processing while registering a new face

    face_names = []  # Clear previous detections
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Error: No frame captured!")
        return

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and extract encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    detected_real_face = False  

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Invalid or Not Registered"  # Default name

        if True in matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            name = known_face_names[best_match_index]
            detected_real_face = True  

        face_names.append(name)

        # Scale back face location
        top, right, bottom, left = [int(i * 2) for i in (top, right, bottom, left)]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Blink Detection
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

    # Update GUI
    welcome.config(text=f"Hi {face_names[0]}! Welcome!" if detected_real_face else "⚠️ Only real humans can be recognized!", fg="black" if detected_real_face else "red")
    recognized_label.config(text="Recognized: " + ", ".join(face_names) if face_names else "Recognized: Unknown")

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
    imgtk = ImageTk.PhotoImage(image=img)
    display1.configure(image=imgtk)
    display1.image = imgtk

    window.after(10, show_frame)  # Repeat every 10 ms

# -----------------------------------------
#  🔹 Face Registration
# -----------------------------------------

def register():
    """
    Registers a new face in the database after confirming liveliness (blink detection).
    """
    global is_registering  

    if blinks_detected < 2:
        messagebox.showwarning("Warning", "Please blink twice to confirm liveliness.")
        return

    is_registering = True  # Prevents multiple registrations

    name = simpledialog.askstring("Register Face", "Please enter your name:")
    if name:
        ret, frame = cap.read()
        if ret:
            face_encoding = face_recognition.face_encodings(frame)[0]  
            encoded_face = pickle.dumps(face_encoding)

            cursor = conn.cursor()
            cursor.execute("INSERT INTO faces (name, encoding) VALUES (%s, %s)", (name, encoded_face))
            conn.commit()
            cursor.close()

            messagebox.showinfo("Success", f"Face registered as {name}!")

    is_registering = False  

# Button for registration
btn = tk.Button(window, text="Register", font="Helvetica 16 bold", fg="black", bg="white", command=register)
btn.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

display1 = tk.Label(imageFrame)
display1.grid(row=2, column=0, columnspan=3, padx=1, pady=2)

show_frame()
window.mainloop()
