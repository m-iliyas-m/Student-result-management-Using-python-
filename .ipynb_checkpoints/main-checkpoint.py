import cv2
import os
from handtrackingmodule import HandDetector
import numpy as np


#Variables
width , height = 1536,864
gestureThreshold = 300
folderPath = "Presentation"


# Camera Setup
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

#get the list of presentation images
pathImages = sorted(os.listdir(folderPath),key=len)
print(pathImages)

# variables
imgNumber = 0
buttonPressed = False
buttoncounter = 0
buttondelay = 10
hs,ws = int(120*1),int(213*1)
annotations = [[]]
annotationNumber = 0
annotationStart = False
fingersLeft = None
fingersRight = None
highlighter_color = (0, 0, 255)
pointer_size = 12

#Hand Detector
detector = HandDetector(detectionCon = 0.8, maxHands=2)

while True:
    #Import Images
    success ,img = cap.read()
    img = cv2.flip(img,1)
    pathFullImage = os.path.join(folderPath,pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)
    img = cv2.resize(img,(imgCurrent.shape[1], imgCurrent.shape[0]))
    hands,img = detector.findHands(img)
    cv2.line(img,(0,gestureThreshold),(width,gestureThreshold),(0,255,0),10)
    left_hand = None
    right_hand = None



    if hands and buttonPressed is False:
        for hand in hands:
            handType = hand["type"]
            if handType == 'Left':
                left_hand = hand
            elif handType == 'Right':
                right_hand = hand
        hand = hands[0]
        hand_landmarks = hands[0]['lmList']  # Landmarks of the first detected hand
        index_tip = hand_landmarks[8]

        if right_hand:
            fingersRight = detector.fingersUp(hand)
        elif left_hand:
            fingersLeft = detector.fingersUp(hand)


        # print(fingers)
        cx,cy = hand['center']
        lmList = hand['lmList']
        # indexFinger = lmList[8][0],lmList[8][1]


        # constrain values for easier drawing
        xVAL = int(np.interp(lmList[8][0], [width//2,width], [0, width]))
        yVAL =int(np.interp(lmList[8][1], [150, height-150], [0, height]))

        indexFinger = xVAL,yVAL

        if cy <= gestureThreshold: # if hand is at the height at the face
            annotationStart = False
            #gesture 1 - left
            if fingersLeft == [1,0,0,0,0]:
                annotationStart = False
                print("left")
                if imgNumber > 0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = 0
                    imgNumber -= 1
            # gesture 2 - right
            if fingersLeft == [0, 0, 0, 0, 1]:
                annotationStart = False
                print("Right ")
                if imgNumber < len(pathImages)-1:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = 0
                    imgNumber += 1

        # gesture 6 colour change
        if fingersRight == [0, 1, 0, 0, 0]:
            highlighter_color = (0, 0, 255)
            annotationStart = False

        # gesture 6 colour change
        if fingersRight == [0, 1, 1, 0, 0]:
            highlighter_color = (0, 255, 255)
            annotationStart = False

            # gesture 6 colour change
        if fingersRight == [0, 1, 1, 1, 0]:
            highlighter_color = (255, 0, 0)
            annotationStart = False

         # gesture 3 - show pointer
        if fingersLeft == [0, 1, 1, 0, 0]:
            # print(highlighter_color)
            cv2.circle(imgCurrent,indexFinger,12,highlighter_color,cv2.FILLED)
            annotationStart = False

        # gesture 4 - Draw pointer
        if fingersLeft == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber +=1
                annotations.append([])
            annotations[annotationNumber].append(indexFinger)
            print(highlighter_color)
            cv2.circle(imgCurrent, indexFinger, pointer_size,color= highlighter_color,thickness=cv2.FILLED)
        else:
            annotationStart = False

        # gesture 5 erase
        if fingersLeft == [0, 1, 1, 1, 0]:
            if annotations:
                    if annotationNumber > -1:
                        annotations.pop(-1)
                        annotationNumber -=1
                        buttonPressed = True

    else:
        annotationStart = False



    #button pressed ittreations
    if buttonPressed:
        buttoncounter +=1
        if buttoncounter > buttondelay :
            buttoncounter = 0
            buttonPressed = False

    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurrent,annotations[i][j-1],annotations[i][j],highlighter_color,12)

    #Adding webcam image on the slides
    imgSmall = cv2.resize(img,(ws,hs))
    h,w, _ = imgCurrent.shape
    imgCurrent[0:hs,w-ws:w] = imgSmall
    cv2.imshow("image",img)
    cv2.imshow("Slides", imgCurrent)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
