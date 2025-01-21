import cv2
import os

# Chemin de la vidéo
video_path = "data/FoxP2_#25_090322_6OHDA_20Hz_GP_comp.MP4"
# Dossier où les frames seront sauvegardées
output_folder = "FoxP2_#25_090322_6OHDA_20Hz_GP_comp"

# Timecodes en secondes (par exemple, 5 minutes = 300 secondes, 6 min 30 = 390 secondes)
start_time = 5 * 60 + 53.5  # 5 minutes 46 (en secondes)
end_time = 5 * 60 + 57  # 6 minutes (en secondes)

# Créez le dossier de sortie s'il n'existe pas
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Chargement de la vidéo
cap = cv2.VideoCapture(video_path)

# Vérifiez si la vidéo est ouverte correctement
if not cap.isOpened():
    print("Impossible d'ouvrir la vidéo.")
    exit()

# Récupération du framerate (fps) pour sauter aux timecodes souhaités
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Nombre d'images par seconde
start_frame = int(start_time * fps)  # Frame correspondant au début
end_frame = int(end_time * fps)  # Frame correspondant à la fin
print(f"Extraction des frames entre {start_time} secondes et {end_time} secondes.")
print(f"Frames correspondant aux timecodes {start_frame} et {end_frame}.")
# Positionnez la vidéo au début du timecode
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frame_count = start_frame
while frame_count <= end_frame:
    ret, frame = cap.read()
    if not ret:  # Fin de la vidéo ou problème de lecture
        break
    
    # Nom de la frame
    frame_name = f"{output_folder}/frame_{frame_count:04d}.jpg"
    # Sauvegarde de la frame
    cv2.imwrite(frame_name, frame)
    frame_count += 1

cap.release()
print(f"Extraction terminée ! Frames extraites entre {start_time} secondes et {end_time} secondes dans le dossier '{output_folder}'.")
