import cv2
import threading
from datetime import datetime
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from ultralytics import YOLO
from .models import FireEvent
import os
import pygame

# Load YOLO model
model = YOLO("best6.pt")

# Initialize pygame mixer once
pygame.mixer.init()
alarm_path = os.path.join("assets", "alarm2.mp3")

# Global video capture
cap = cv2.VideoCapture(0)

# Play alarm sound in a new thread
def play_alarm():
    def sound_thread():
        try:
            pygame.mixer.music.load(alarm_path)
            pygame.mixer.music.play()
        except Exception as e:
            print("Sound error:", e)

    threading.Thread(target=sound_thread, daemon=True).start()

# Fire detection frame generator
def generate_frames():
    fire_detected = False
    start_time = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, conf=0.5, iou=0.5, verbose=False)
        fire_in_frame = any(r.boxes.shape[0] > 0 for r in results)

        if fire_in_frame and not fire_detected:
            fire_detected = True
            start_time = datetime.now()
            play_alarm()
        elif not fire_in_frame and fire_detected:
            fire_detected = False
            end_time = datetime.now()
            duration = end_time - start_time
            FireEvent.objects.create(detected_at=start_time, duration=duration)

        annotated_frame = results[0].plot()
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Views
def index(request):
    logs = FireEvent.objects.all().order_by('-detected_at')
    return render(request, 'detector/index.html', {'logs': logs})

@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(generate_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def reset_logs(request):
    FireEvent.objects.all().delete()
    return redirect('index')
