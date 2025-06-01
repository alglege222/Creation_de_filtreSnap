import cv2
import numpy as np

nose_img = cv2.imread("nose.png", cv2.IMREAD_UNCHANGED)
ear_left_img = cv2.imread("ear-right.png", cv2.IMREAD_UNCHANGED)
ear_right_img = cv2.imread("ear-left.png", cv2.IMREAD_UNCHANGED)

nose_img_color = nose_img[:, :, :3]
nose_img_alpha = nose_img[:, :, 3]
ear_left_img_color = ear_left_img[:, :, :3]
ear_left_img_alpha = ear_left_img[:, :, 3]
ear_right_img_color = ear_right_img[:, :, :3]
ear_right_img_alpha = ear_right_img[:, :, 3]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Lancer le webcam
cap = cv2.VideoCapture(0)

nose_width = 100
nose_height = 100
ears_width = 150
ears_height = 150

while True:
    
    ret, frame = cap.read()

    if not ret:
        break

    
    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    
    for (x, y, w, h) in faces:
        
        face_roi = frame_rgba[y:y+h, x:x+w]

        
        nose_img_resized_color = cv2.resize(nose_img_color, (nose_width, nose_height))
        nose_img_resized_alpha = cv2.resize(nose_img_alpha, (nose_width, nose_height))
        ear_left_img_resized_color = cv2.resize(ear_left_img_color, (ears_width, ears_height))
        ear_left_img_resized_alpha = cv2.resize(ear_left_img_alpha, (ears_width, ears_height))
        ear_right_img_resized_color = cv2.resize(ear_right_img_color, (ears_width, ears_height))
        ear_right_img_resized_alpha = cv2.resize(ear_right_img_alpha, (ears_width, ears_height))

        
        nose_img_resized = np.dstack((nose_img_resized_color, nose_img_resized_alpha))
        ear_left_img_resized = np.dstack((ear_left_img_resized_color, ear_left_img_resized_alpha))
        ear_right_img_resized = np.dstack((ear_right_img_resized_color, ear_right_img_resized_alpha))

        
        nose_top_left = (int(x + w/2 - nose_width/2), int(y + h/2 - nose_height/2))
        ear_left_top_left = (int(x + w/4 - ears_width/2), int(y - ears_height/2))
        ear_right_top_left = (int(x + 3*w/4 - ears_width/2), int(y - ears_height/2))

        
        for i in range(nose_img_resized.shape[0]):
            for j in range(nose_img_resized.shape[1]):
                if nose_img_resized[i, j, 3] != 0:
                    frame_rgba[nose_top_left[1] + i, nose_top_left[0] + j, :3] = nose_img_resized[i, j, :3]

        for i in range(ear_left_img_resized.shape[0]):
            for j in range(ear_left_img_resized.shape[1]):
                if ear_left_img_resized[i, j, 3] != 0:
                    frame_rgba[ear_left_top_left[1] + i, ear_left_top_left[0] + j, :3] = ear_left_img_resized[i, j, :3]

        for i in range(ear_right_img_resized.shape[0]):
            for j in range(ear_right_img_resized.shape[1]):
                if ear_right_img_resized[i, j, 3] != 0:
                    frame_rgba[ear_right_top_left[1] + i, ear_right_top_left[0] + j, :3] = ear_right_img_resized[i, j, :3]


    frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)


    cv2.imshow("Snapchat Filter", frame_bgr)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()