import cv2
import mediapipe as mp
# tensorflow==2.3.0
from tensorflow.keras.models import load_model
import numpy as np
import copy
import itertools
from datetime import datetime,timedelta

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

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


model = load_model("sampleModel")

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

labels_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
               'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
               'X': 23, 'Y': 24, 'Z': 25, 'space': 26, 'del': 27, 'nothing': 28}


def get_key(val):
    for key, value in labels_dict.items():
        if val == value:
            return key
    return "nothing"

class_x_1 =[]
tempList =[]
startTime = datetime.now()

while True:
    success, image = cap.read()
    # image = cv2.imread("5.png", cv2.IMREAD_COLOR)
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    unique_list=[]
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            landmark_list = calc_landmark_list(imgRGB, hand_landmarks)
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            predictions = model.predict(np.array([pre_processed_landmark_list], dtype=np.float32))
            classes_x = np.argmax(np.squeeze(predictions))
            key = get_key(classes_x)
            
            tempList.append(key)
            if len(set(tempList)) != 1:
                tempList = []
                startTime = datetime.now()
                
            if int((datetime.now() - startTime).total_seconds()) > 1: #hold the hand sign for at-least 1s
                if key:
                    if key == "space":
                        class_x_1.append(" ")
                    elif key == "del":
                        if (len(class_x_1) > 0):
                            class_x_1.pop()
                    elif key != "nothing":
                        class_x_1.append(key)
                tempList = []
                startTime = datetime.now()
            
            print("\nLabel")
            print(key)
            print("\n")

            cv2.putText(image,''.join(class_x_1), (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 4)

    cv2.imshow("Results", image)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break


