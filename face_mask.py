import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define a function to detect faces based on simple color thresholding
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Dummy face detection using color thresholds; replace with real detection logic
    faces = []
    height, width = gray.shape
    # Assuming face region as a center rectangle for demonstration
    x, y, w, h = int(width/4), int(height/4), int(width/2), int(height/2)
    faces.append((x, y, w, h))
    return faces

# Define a function to check for masks
def detect_mask(face_image):
    # Simple mask detection logic based on color or shape (dummy example)
    # Note: This is highly inaccurate and for demonstration only
    mask_detected = False
    if np.mean(face_image) < 100:  # Dummy threshold for mask detection
        mask_detected = True
    return mask_detected

# Main loop for live camera feed
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    faces = detect_faces(frame)

    # Draw bounding boxes and check for masks
    for (x, y, w, h) in faces:
        face_image = frame[y:y+h, x:x+w]
        mask_detected = detect_mask(face_image)
        color = (0, 255, 0) if mask_detected else (0, 0, 255)
        label = 'Mask' if mask_detected else 'No Mask'
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the resulting frame
    cv2.imshow('Face Mask Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
