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
import pickle  # Used to serialize the face encodings

# Eye aspect ratio (EAR) calculation function for blink detection
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Indices for eye landmarks
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# Blink detection parameters
BLINK_THRESHOLD = 0.25  # Eye aspect ratio threshold for blink detection
BLINK_FRAMES = 1  # Minimum frames required to consider it as a blink
blink_counter = 0
blinks_detected = 0  # To count how many blinks are detected

# Load face detector and shape predictor
PREDICTOR_PATH = os.path.join(os.getcwd(), "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

# MySQL database connection setup
def init_db():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",  # Change this to your MySQL username
        password="123456",  # Change this to your MySQL password
        database="face_recognition_db"  # Ensure you've created this database
    )
    return conn

# Function to load known faces from the MySQL database
def load_known_faces(conn):
    cursor = conn.cursor()
    known_face_encodings = []
    known_face_names = []
    
    cursor.execute("SELECT name, encoding FROM faces")
    rows = cursor.fetchall()
    
    for row in rows:
        name = row[0]
        encoding = pickle.loads(row[1])  # Convert from BLOB to numpy array
        known_face_names.append(name)
        known_face_encodings.append(encoding)
    
    cursor.close()
    return known_face_encodings, known_face_names

# Create log file for tracking registrations
file_name = "logs" + str(datetime.now())[:10] + ".txt"
if not os.path.exists(file_name):
    f = open(file_name, 'a+')
    f.close()

# Set up GUI
window = tk.Tk()
window.wm_title("Face Recognition for Access Control")
window.config(background="#74b0c0")
font = cv2.FONT_HERSHEY_SIMPLEX

# Create Graphics window for the camera feed
imageFrame = tk.Frame(window, width=600, height=600, bg="white")
imageFrame.grid(row=0, column=0, rowspan=3, padx=10, pady=2)

# Capture video frames
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Camera not detected! Exiting...")
    exit()

# Initialize the main label where the recognized face will be shown
recognized_label = tk.Label(window, text="Recognized: None", font="Helvetica 16 bold", fg="black", bg="white")
recognized_label.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

# Initialize the welcome message label
welcome = tk.Label(window, text="                               ", font="Helvetica 16 bold", fg="white", bg="SteelBlue4")
welcome.grid(row=2, column=1, padx=5, pady=5)

# Declare face_names as a global variable (moving it out of the show_frame function)
face_names = []

# Initialize the MySQL database connection
conn = init_db()

# Load known faces from the MySQL database
known_face_encodings, known_face_names = load_known_faces(conn)

# Function to show video feed and handle face detection
def show_frame():
    global blink_counter, blinks_detected, face_names  # Declare global face_names
    face_names = []  # Reset the list each time the frame is processed
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Error: No frame captured!")
        return

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            name = known_face_names[best_match_index]
        face_names.append(name)

        # Draw rectangle around the face
        top = int(top * 2)
        right = int(right * 2)
        bottom = int(bottom * 2)
        left = int(left * 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Convert to grayscale for dlib landmark detection
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

            # Check for blink detection
            if avg_EAR < BLINK_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= BLINK_FRAMES:
                    blinks_detected += 1
                blink_counter = 0

    # Update recognized label
    recognized_label.config(text="Recognized: " + ", ".join(face_names) if face_names else "Recognized: Unknown")

    # Convert frame to image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
    imgtk = ImageTk.PhotoImage(image=img)

    # Display frame with rectangle
    display1.configure(image=imgtk)
    display1.image = imgtk  # Important to keep a reference to the image

    window.after(10, show_frame)  # Call the function again after a short delay

# Function to handle registration of a new face
def register(window, welcome):
    global blink_counter, blinks_detected, face_names  # Declare global face_names
    if blinks_detected >= 2:
        for name in face_names:
            if name == "Unknown":
                # Prompt for user input to set a name for the face
                name = simpledialog.askstring("Register Face", "Please enter your name:")
                if name:
                    print(f"Registering {name}")
                    # Add the new face encoding to the known face encodings and names list
                    ret, frame = cap.read()
                    if ret:
                        # Save face encoding to the MySQL database
                        face_encoding = face_recognition.face_encodings(frame)[0]
                        encoded_face = pickle.dumps(face_encoding)  # Serialize encoding to BLOB
                        cursor = conn.cursor()
                        cursor.execute("INSERT INTO faces (name, encoding) VALUES (%s, %s) ON DUPLICATE KEY UPDATE encoding=%s", 
                                       (name, encoded_face, encoded_face))
                        conn.commit()
                        cursor.close()

                        # Log the registration
                        with open(file_name, 'a+') as f:
                            f.write(f"{str(datetime.now())[11:-10]}\t{name}\n")

                        welcome.config(text="Hi " + name.capitalize())
                        messagebox.showinfo("Success", f"Face registered as {name}!")
    else:
        messagebox.showwarning("Warning", "Please blink twice to confirm liveliness.")

# Create button to register when user confirms with blink
btn = tk.Button(window, text="Register", font="Helvetica 16 bold", fg="white", bg="SteelBlue4",
                command=lambda: register(window, welcome))
btn.grid(row=0, column=1, padx=5, pady=5)

# Display camera feed
display1 = tk.Label(imageFrame)
display1.grid(row=1, column=0, columnspan=3, padx=10, pady=2)

# Start the frame processing
show_frame()

# Run the GUI
window.mainloop()
