# FaceRecognition-main
 This project implements a face recognition system for access control using a camera, MySQL database for storing face data, and a simple GUI built with Tkinter. The system recognizes registered faces and logs events when a face is recognized.

**Features**
Face Detection: Detects faces in real-time using dlib and face_recognition.
Blink Detection: Verifies liveliness with blink detection using the Eye Aspect Ratio (EAR).
MySQL Integration: Stores face encodings in a MySQL database.
GUI: A user-friendly interface built with Tkinter to display video feed, recognized faces, and provide options for face registration.
Logging: Logs the name and timestamp of registered faces.
**Requirements**
**Dependencies:**
Python 3.x
MySQL
Required libraries:
mysql-connector-python
face_recognition
dlib
numpy
opencv-python
Pillow
tkinter
scipy
<pre>CREATE DATABASE face_recognition_db;

USE face_recognition_db;

CREATE TABLE faces (
    name VARCHAR(255) PRIMARY KEY,
    encoding BLOB
);
<pre>
**Notes**
The system recognizes faces by comparing their encodings to the ones stored in the database.
The database stores each face’s name and encoding as a BLOB. The encodings are serialized using pickle.
The system requires a camera for face detection.

**Customize:**
Replace placeholders: Ensure that you replace the placeholders like your_mysql_user and your_mysql_password in the script with actual credentials.
Pre-trained Model: Make sure the shape_predictor_68_face_landmarks.dat file is downloaded and available in your project directory.
