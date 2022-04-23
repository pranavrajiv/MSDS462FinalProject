import cv2
import mediapipe as mp
import time
from tensorflow import keras
from PIL import Image
import numpy as np

#model located at https://www.kaggle.com/code/sayakdasgupta/sign-language-classification-cnn-99-40-accuracy/data
model = keras.models.load_model('asl_predictor.h5')

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
        for handLms in results.multi_hand_landmarks:
            h, w, c = image.shape
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for id, lm in enumerate (handLms.landmark):
                #print(id,lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                #print(id, cx, cy)
                if id == 4:
                    cv2.circle(image,(cx,cy),15, (255,0,255),cv2.FILLED)

                if cx > x_max:
                    x_max = cx
                if cx < x_min:
                    x_min = cx
                if cy > y_max:
                    y_max = cy
                if cy < y_min:
                    y_min = cy

            cv2.rectangle(image, (x_min - 50, y_min - 50), (x_max + 50, y_max + 50), (0, 255, 0), 2)
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
            cropped_image = image[y_min - 50:y_max - 50, x_min + 50:x_max + 50]
            #cv2.imwrite('savedImage.jpg', cropped_image)
            
            size = 64,64
            i = cv2.resize(cropped_image, size)
            i = i.astype('float32')/255.0
            predictions = model.predict(i.reshape(1,64,64,3))
            classes_x=np.argmax(predictions,axis=1)
            print("\nHello\n")
            print(get_key(classes_x[0]))
            cv2.putText(image, get_key(classes_x[0]),(10,60), cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),4)
            print("\nHello\n")
            
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

