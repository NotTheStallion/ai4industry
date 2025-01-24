import os
import numpy as np
import json
from reservoirpy.nodes import ESN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
from sklearn.decomposition import PCA

# Chemin du dossier contenant les frames de flux optique
frames_folder = "/home/cbanide/ai4industry/test_flows/farneback_optical_flow"

# Charger les annotations
annotations_file = "/home/cbanide/ai4industry/annotations.json"
with open(annotations_file, "r") as f:
    annotations = json.load(f)

# Charger les frames et leurs labels
X = []
Y = []

for frame_name, label in annotations.items():
    frame_path = os.path.join(frames_folder, frame_name)
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if frame is not None:
        frame = cv2.resize(frame, (64, 64))  # Réduction de la taille à 64x64
        X.append(frame.flatten())
        Y.append(label)
    else:
        print(f"Impossible de charger : {frame_path}")

X = np.array(X)
Y = np.array(Y)

# Vérifications des données
print(f"Dimensions de X : {X.shape}")
if len(X) == 0:
    raise ValueError("Aucune donnée valide trouvée dans X. Vérifiez vos fichiers et annotations.")

# Option : Réduction de la dimension avec PCA
pca = PCA(n_components=100)  # Réduction à 100 dimensions
X = pca.fit_transform(X)
print(f"Dimensions de X après PCA : {X.shape}")

# Convertir les labels en valeurs numériques
unique_labels = list(set(Y))
label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
num_to_label = {idx: label for label, idx in label_to_num.items()}
Y_numeric = np.array([label_to_num[label] for label in Y])

# Vérifier les dimensions des données et des labels
print(f"Dimensions de X : {X.shape}")
print(f"Dimensions de Y : {Y_numeric.shape}")
print(f"Exemple de Y : {Y_numeric[:5]}")

# Reshape des labels si nécessaire
if Y_numeric.ndim == 1:
    Y_numeric = Y_numeric.reshape(-1, 1)

print(f"Dimensions corrigées de Y : {Y_numeric.shape}")

# Vérifier les classes dans Y
print(f"Classes uniques dans Y : {np.unique(Y_numeric)}")
if len(np.unique(Y_numeric)) < 2:
    raise ValueError("Y doit contenir au moins deux classes pour l'entraînement.")

# Créer le modèle ReservoirPY
esn = ESN(
    n_inputs=X.shape[1],  # Taille de l'entrée après PCA
    n_reservoir=500,     # Nombre de neurones dans le réservoir
    units=500,           # Définit explicitement les unités (même que n_reservoir)
    spectral_radius=0.9, # Contrôle la dynamique du réservoir
    sparsity=0.1,        # Connectivité du réservoir
    ridge=1e-6,          # Régularisation pour éviter les matrices mal conditionnées
    random_state=42
)

# Entraîner le modèle sur l'ensemble complet des données
esn.fit(X, Y_numeric)

# Tester sur l'ensemble complet des données
Y_pred = esn.run(X)

# Convertir les prédictions pour correspondre aux classes
Y_pred = np.argmax(Y_pred, axis=1)  # Si Y_pred est un tableau de probabilités

# Évaluer les performances
accuracy = accuracy_score(Y_numeric.flatten(), Y_pred)
print("Précision du modèle sur l'ensemble complet des données :", accuracy)

# Afficher les prédictions et leurs labels
for idx, pred in enumerate(Y_pred):
    print(f"Frame {idx + 1}: Prédit = {num_to_label[pred]}, Réel = {num_to_label[Y_numeric[idx][0]]}")
