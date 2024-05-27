import torch
import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import os
import threading

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define a function for text-to-speech conversion
def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = 'temp.mp3'
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

# Define a function to give audio alerts based on detected objects
def give_alerts(detections):
    for det in detections:
        label = det['label']
        confidence = det['confidence']
        if confidence > 0.5:
            if label in ["car", "truck", "bus", "motorbike", "bicycle"]:
                speak(f"Attention! {label} detected. Please be cautious of vehicles around you.")
            elif label == "traffic light":
                speak("Traffic light ahead. Please check the light status.")
            elif label == "person":
                speak("Person detected ahead. Be careful.")
            elif label == "stop sign":
                speak("Stop sign ahead. Be prepared to stop.")
            elif label == "crosswalk":
                speak("Crosswalk detected. You may proceed if it's safe.")
            elif label == "dog":
                speak("Dog detected ahead. Please be cautious.")
            elif label == "cat":
                speak("Cat detected ahead. Please be cautious.")
            elif label == "bicycle":
                speak("Bicycle detected ahead. Please be cautious.")
            elif label == "motorbike":
                speak("Motorbike detected ahead. Please be cautious.")
            elif label == "bus":
                speak("Bus detected ahead. Please be cautious.")
            elif label == "truck":
                speak("Truck detected ahead. Please be cautious.")
            else:
                speak(f"Alert! {label} detected ahead. Please be aware of your surroundings.")

# Function to process frame and provide alerts
def process_frame(frame):
    results = model(frame)
    detections = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        label = model.names[int(cls)]
        detections.append({'label': label, 'confidence': conf, 'box': box})
    
    for det in detections:
        label = det['label']
        confidence = det['confidence']
        x1, y1, x2, y2 = map(int, det['box'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    give_alerts(detections)

# Open a video capture (change the path to your video file)
video_path = 'path_to_your_video.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Only process every 3rd frame to reduce load
    if frame_count % 3 != 0:
        continue

    # Resize frame to reduce processing load
    frame = cv2.resize(frame, (640, 480))
    
    # Create a thread to process the frame and provide alerts
    thread = threading.Thread(target=process_frame, args=(frame,))
    thread.start()
    
    # Display the frame
    cv2.imshow('Assistive Application', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
