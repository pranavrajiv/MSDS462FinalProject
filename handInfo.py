import cv2
import mediapipe as mp
import time
from tensorflow import keras # tensorflow==2.3.0
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import copy
import itertools

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

#model located at https://www.kaggle.com/code/sayakdasgupta/sign-language-classification-cnn-99-40-accuracy/data
model = load_model('newModel.hdf5')

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


pTime = 0
cTime = 0


labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25,'space':26,'del':27,'nothing':28}


def get_key(val):
    for key, value in labels_dict.items():
         if val == value:
             return key

    return "key doesn't exist"



while True:
    success, image = cap.read()
    #image = cv2.imread("5.png", cv2.IMREAD_COLOR)
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
            h, w, c = image.shape
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            #for id, lm in enumerate (handLms.landmark):
            landmark_list = calc_landmark_list(imgRGB, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            #cropped_image = image[y_min - 100:y_max + 100, x_min - 100:x_max + 100]

            predictions = model.predict(np.array([pre_processed_landmark_list], dtype=np.float32))
            print("\nHeloo\n")
            classes_x=np.argmax(np.squeeze(predictions))
            #cv2.rectangle(image, (x_min - 100, y_min - 100), (x_max + 100, y_max + 100), (0, 255, 0), 2)
            #mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
            print(get_key(classes_x))
            cv2.putText(image, get_key(classes_x),(10,60), cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),4)
            print("\nLabelo\n")

            #predictions = [model.predict(cropped_image.reshape(1,64,64,3))]
#    if results.multi_hand_world_landmarks:
#        for handLms in results.multi_hand_world_landmarks:
#            for id, lm in enumerate (handLms.landmark):
#                print(id,lm)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #cv2.putText(image, str(int(fps)),(10,60), cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),4)
    cv2.imshow("Results", image)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

