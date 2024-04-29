import cv2
from cvzone.HandTrackingModule import HandDetector # type: ignore
import numpy as np
import math


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)


offset=20
imgsize=300

while True:
    success, img = cap.read()
    hands, img  = detector.findHands(img)
    if hands:
        hand1 = hands[0]
        bbox1 = hand1["bbox"]
        
        imgWhite=np.ones((imgsize,imgsize,3),np.uint8)*255
        
        img2 = img[bbox1[1]:bbox1[1]+bbox1[3], bbox1[0]:bbox1[0]+bbox1[2]]
        
        imgcrop = img2.shape
        
        aspect_ratio=bbox1[3]/bbox1[2]  
        
        """
        if aspect_ratio>1:
            k=imgsize/bbox1[3]
            wcal =  math.ceil(k*bbox1[2])
            img3 = cv2.resize(img2, (wcal,imgsize))
            imgResizeShape=img3.shape
            wGap = math.ceil((imgsize-wcal)/2)
            imgWhite[0:wGap:wcal+wGap] = img3
        else:
            k=imgsize/bbox1[2]
            hcal =  math.ceil(k*bbox1[3])
            img3 = cv2.resize(img2, (imgsize,hcal))
            imgResizeShape=img3.shape
            hGap = math.ceil((imgsize-hcal)/2)
            #imgWhite[hGap:hcal+hGap,0:imgsize] = img3"""
        
        cv2.imshow("ImageCrop", img2)
        cv2.imshow("ImageWhite", imgWhite)
        
    cv2.imshow("Image", img)
    cv2.waitKey(1)