# Operation de  traitement d'image
import cv2
# Utiliser pour les opérations sur les tableaux (arrays).
import numpy as np

# Ces lignes chargent les images
nose_img = cv2.imread("C:\\Users\\DELL\\OneDrive\\Documents\\LICENCE_2_DIT_BIG_DATA\\Creation_filtre\\snap filters\\nose.png", cv2.IMREAD_UNCHANGED)
ear_left_img = cv2.imread("C:\\Users\\DELL\\OneDrive\\Documents\\LICENCE_2_DIT_BIG_DATA\\Creation_filtre\\snap filters\\ear-right.png", cv2.IMREAD_UNCHANGED)
ear_right_img = cv2.imread("C:\\Users\\DELL\\OneDrive\\Documents\\LICENCE_2_DIT_BIG_DATA\\Creation_filtre\\snap filters\\ear-left.png", cv2.IMREAD_UNCHANGED)

# Ces lignes séparent les images en deux parties: les trois premiers canaux (couleur) et le quatrième canal (alpha). Cela permet de manipuler séparément les informations de couleur et de transparence.


nose_img_color = nose_img[:, :, :3]
nose_img_alpha = nose_img[:, :, 3]
ear_left_img_color = ear_left_img[:, :, :3]
ear_left_img_alpha = ear_left_img[:, :, 3]
ear_right_img_color = ear_right_img[:, :, :3]
ear_right_img_alpha = ear_right_img[:, :, 3]

# Cette ligne charge le classificateur de visage
# un modèle pré-entraîné pour la détection des visages basé sur l'algorithme Haar.


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Lancer le webcam
# Cette ligne initialise la capture vidéo à partir de la webcan

cap = cv2.VideoCapture(0)

# Ces lignes définissent les dimensions souhaitées pour les images
nose_width = 100
nose_height = 100
ears_width = 150
ears_height = 150

# La boucle while lit en continu les images de la webcam.

while True:
    
    ret, frame = cap.read()

    if not ret:
        break

    # Cette ligne convertit l'image de la webcam de BGR à RGBA pour inclure le canal alpha.

    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    # Cette ligne convertit l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Cette ligne détecte les visages dans l'image en niveaux de gris
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Cette boucle parcourt chaque visage détecté.Pour repeere les regions utiles du visage
    for (x, y, w, h) in faces:
        
        face_roi = frame_rgba[y:y+h, x:x+w]

        # Ces lignes redimensionnent les images du nez et des oreilles à leurs dimensions souhaitées.

        nose_img_resized_color = cv2.resize(nose_img_color, (nose_width, nose_height))
        nose_img_resized_alpha = cv2.resize(nose_img_alpha, (nose_width, nose_height))
        ear_left_img_resized_color = cv2.resize(ear_left_img_color, (ears_width, ears_height))
        ear_left_img_resized_alpha = cv2.resize(ear_left_img_alpha, (ears_width, ears_height))
        ear_right_img_resized_color = cv2.resize(ear_right_img_color, (ears_width, ears_height))
        ear_right_img_resized_alpha = cv2.resize(ear_right_img_alpha, (ears_width, ears_height))

        # Ces lignes empilent les canaux de couleur et alpha pour recréer des images RGBA redimensionnées.

        nose_img_resized = np.dstack((nose_img_resized_color, nose_img_resized_alpha))
        ear_left_img_resized = np.dstack((ear_left_img_resized_color, ear_left_img_resized_alpha))
        ear_right_img_resized = np.dstack((ear_right_img_resized_color, ear_right_img_resized_alpha))

        # Ces lignes calculent les positions supérieures gauches pour placer les images du nez et des oreilles sur le visage détecté.

        nose_top_left = (int(x + w/2 - nose_width/2), int(y + h/2 - nose_height/2))
        ear_left_top_left = (int(x + w/4 - ears_width/2), int(y - ears_height/2))
        ear_right_top_left = (int(x + 3*w/4 - ears_width/2), int(y - ears_height/2))

        # Cette double boucle parcourt chaque pixel de l'image du nez redimensionnée. Si le pixel n'est pas transparent (alpha != 0), il est superposé sur le frame RGBA.
        for i in range(nose_img_resized.shape[0]):
            for j in range(nose_img_resized.shape[1]):
                if nose_img_resized[i, j, 3] != 0:
                    frame_rgba[nose_top_left[1] + i, nose_top_left[0] + j, :3] = nose_img_resized[i, j, :3]
        # Cette boucle fait la même chose pour l'oreille gauche
        for i in range(ear_left_img_resized.shape[0]):
            for j in range(ear_left_img_resized.shape[1]):
                if ear_left_img_resized[i, j, 3] != 0:
                    frame_rgba[ear_left_top_left[1] + i, ear_left_top_left[0] + j, :3] = ear_left_img_resized[i, j, :3]
        # Cette boucle fait la même chose pour l'oreille droite
        for i in range(ear_right_img_resized.shape[0]):
            for j in range(ear_right_img_resized.shape[1]):
                if ear_right_img_resized[i, j, 3] != 0:
                    frame_rgba[ear_right_top_left[1] + i, ear_right_top_left[0] + j, :3] = ear_right_img_resized[i, j, :3]

    # Cette ligne convertit l'image de RGBA à BGR pour l'affichage avec OpenCV.
    frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)

    # Cette ligne affiche l'image avec les filtres appliqués dans une fenêtre
    cv2.imshow("Snapchat Filter", frame_bgr)

    # Cette ligne attend 1 ms pour une pression de touche. Si la touche 'q' est pressée, la boucle se termine.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ces lignes libèrent la capture vidéo et ferment toutes les fenêtres OpenCV ouvertes.
cap.release()
cv2.destroyAllWindows()