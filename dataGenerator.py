import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import copy
import itertools
import os
import csv

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

def get_key(val):
    for key, value in labels_dict.items():
         if val == value:
             return key

    return "key doesn't exist"


labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25,'space':26,'del':27,'nothing':28}

mpHands = mp.solutions.hands
hands = mpHands.Hands()

with open('data.csv', 'w', newline='') as file:
    for lableName in labels_dict:
        print("\nHello \n"+lableName)
        for filename in os.listdir(lableName):
            f = os.path.join(lableName, filename)
            image = cv2.imread(f, cv2.IMREAD_COLOR)
            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)
            print(f)
            pre_processed_landmark_list = []
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
                    landmark_list = calc_landmark_list(imgRGB, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    pre_processed_landmark_list = np.array([pre_processed_landmark_list], dtype=np.float32)
                    pre_processed_landmark_list = np.insert(pre_processed_landmark_list[0],0,labels_dict.get(lableName), axis=0)
                mywriter = csv.writer(file, delimiter=',')
                mywriter.writerows([pre_processed_landmark_list])
            #break
        #break