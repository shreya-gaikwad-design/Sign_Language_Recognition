import cv2
import numpy as np
import math
import time
import os
#modules to detect hand
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)  #detect only one hand at a time 

offset = 20 
#fixed size for white box 
imgSize = 300




#directory to save images

folder = "images/W"
counter = 0

os.makedirs(folder,exist_ok= True)
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        #dimension for 2nd box 
        x,y,w,h =  hand['bbox']

        #white box to get fixed dimension   
        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255

        imgCrop = img[y-offset: y + h+offset , x -offset: x + w + offset]

        imgCropShape = imgCrop.shape

        #overlaying white box and image
        #height                     #width 
       # imgWhite[0:imgCropShape[0],0:imgCropShape[1]] = imgCrop
        aspectratio=h/w
        #width
        if aspectratio > 1:
            k=imgSize/h
            wcal=math.ceil(k+w)
            imgresize=cv2.resize(imgCrop,(wcal,imgSize))
            imgresizeshape=imgresize.shape
            wgap=math.ceil((imgSize-wcal)/2)
            imgWhite[: ,wgap:wcal+wgap]= imgresize
        

        else:
            k=imgSize/w
            hcal=math.ceil(k+h)
            imgresize=cv2.resize(imgCrop,(imgSize, hcal))
            imgresizeshape=imgresize.shape
            hgap=math.ceil((imgSize-hcal)/2)
            imgWhite[hgap:hcal+hgap, :]= imgresize



        cv2.imshow("ImageCrop ", imgCrop)  #show crop image
        cv2.imshow("ImageWhite ", imgWhite)   #show white box 
    cv2.imshow("Image ", img)
    key1 = cv2.waitKey(1)
    if key1 == ord("s"):
            counter = counter + 1
            timestamp = str(int(time.time()))
            cv2.imwrite(f'{folder}/Image_{timestamp}.jpg',imgWhite)
            print(counter)    
        