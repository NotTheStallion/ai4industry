import os
import numpy as np
import json
from reservoirpy.nodes import ESN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
from sklearn.decomposition import PCA

# doss frame
frames_folder = "/home/cbanide/ai4industry/test_flows/farneback_optical_flow"

annotations_file = "/home/cbanide/ai4industry/annotations.json"
with open(annotations_file, "r") as f:
    annotations = json.load(f)

X = []
Y = []

for frame_name, label in annotations.items():
    frame_path = os.path.join(frames_folder, frame_name)
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if frame is not None:
        frame = cv2.resize(frame, (64, 64))  
        X.append(frame.flatten())
        Y.append(label)
    else:
        print(f"Impossible de charger : {frame_path}")

X = np.array(X)
Y = np.array(Y)

print(f"Dimensions de X : {X.shape}")
if len(X) == 0:
    raise ValueError("Aucune donnée valide trouvée dans X. Vérifiez vos fichiers et annotations.")

pca = PCA(n_components=100)  
X = pca.fit_transform(X)
print(f"Dimensions de X après PCA : {X.shape}")

unique_labels = list(set(Y))
label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
num_to_label = {idx: label for label, idx in label_to_num.items()}
Y_numeric = np.array([label_to_num[label] for label in Y])

print(f"Dimensions de X : {X.shape}")
print(f"Dimensions de Y : {Y_numeric.shape}")
print(f"Exemple de Y : {Y_numeric[:5]}")

if Y_numeric.ndim == 1:
    Y_numeric = Y_numeric.reshape(-1, 1)

print(f"Dimensions corrigées de Y : {Y_numeric.shape}")

print(f"Classes uniques dans Y : {np.unique(Y_numeric)}")
if len(np.unique(Y_numeric)) < 2:
    raise ValueError("Y doit contenir au moins deux classes pour l'entraînement.")

esn = ESN(
    n_inputs=X.shape[1],  
    n_reservoir=500,     
    units=500,           
    spectral_radius=0.9, 
    sparsity=0.1,        
    ridge=1e-6,          
    random_state=42
)

esn.fit(X, Y_numeric)

Y_pred = esn.run(X)

Y_pred = np.argmax(Y_pred, axis=1)  

accuracy = accuracy_score(Y_numeric.flatten(), Y_pred)
print("Précision du modèle sur l'ensemble complet des données :", accuracy)

for idx, pred in enumerate(Y_pred):
    print(f"Frame {idx + 1}: Prédit = {num_to_label[pred]}, Réel = {num_to_label[Y_numeric[idx][0]]}")
