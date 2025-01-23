import cv2
import os

video_path = os.path.abspath('./test.mp4')  

cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Erreur : Impossible de lire la vidéo avec FFmpeg.")
else:
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS de la vidéo : {fps}")

cap.release()
