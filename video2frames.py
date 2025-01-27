import cv2
import os

video_path = "data/test.mp4"
output_folder = "test"

# Timecodes en secondes (par exemple, 5 minutes = 300 secondes, 6 min 30 = 390 secondes)
start_time = 10  # 5 minutes 46 (en secondes)
end_time = 17  # 6 minutes (en secondes)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Impossible d'ouvrir la vidéo.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))  
start_frame = int(start_time * fps)  
end_frame = int(end_time * fps)  
print(f"Extraction des frames entre {start_time} secondes et {end_time} secondes.")
print(f"Frames correspondant aux timecodes {start_frame} et {end_frame}.")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_count = start_frame
while frame_count <= end_frame:
    ret, frame = cap.read()
    if not ret:  
        break
    
    frame_name = f"{output_folder}/frame_{frame_count:04d}.jpg"
    cv2.imwrite(frame_name, frame)
    frame_count += 1

cap.release()
print(f"Extraction terminée ! Frames extraites entre {start_time} secondes et {end_time} secondes dans le dossier '{output_folder}'.")
