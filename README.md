# Face Recognition Attendance System

A **real-time face recognition based attendance system** built with Python, OpenCV, and `face_recognition` library.  
It detects multiple faces at once, marks attendance in a CSV file with **Name, Time, and Date**,  
and displays recognized faces with a **green box** and unknown faces with a **red box**.

---

##  Features
- Detects and recognizes **multiple faces simultaneously**.
- **Green box** for known faces, **Red box** for unknown faces.
- Automatically marks **Name, Time, and Date** in `Attendance.csv`.
- High accuracy using multiple training images per person.
- Live webcam feed inside a **custom UI frame**.

## Technologies Used
- Python 3
- OpenCV
- face_recognition library
- NumPy

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/face-recognition-attendance-system.git
cd face-recognition-attendance-system 
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Prepare training images
- Inside the Images folder, create one folder per person.
- Place multiple images of the person in that folder for better accuracy.

## Run the Project
```bash
python main.py
```
