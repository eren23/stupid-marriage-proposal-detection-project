from numpy.lib.financial import rate
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
import mediapipe as mp
import time

mpPose=mp.solutions.pose
pose = mpPose.Pose()
mpDraw=mp.solutions.drawing_utils


# load model
model = load_model('gender_detection.model')

# open webcam
webcam = cv2.VideoCapture(0)
    
classes = ['man','woman']

# loop through frames
while webcam.isOpened():

    # read frame from webcam 
    status, frame = webcam.read()

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # apply face detection
    face, confidence = cv.detect_face(frame)


    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        middleOne = (int((startX+endX)/2),int((startY+endY)/2))
        

        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        cv2.circle(frame, middleOne, 5,(0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    results = pose.process(imgRGB)
    leftLeg = {} # 23,25,27
    rightLeg = {} #24,26,28

    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = frame.shape
            # print(id,lm)
            cx,cy=int(lm.x*w),int(lm.y*h)
            
            leftLeg[23] = np.array([results.pose_landmarks.landmark[23].x,results.pose_landmarks.landmark[23].y, results.pose_landmarks.landmark[23].z])
            leftLeg[25] = np.array([results.pose_landmarks.landmark[25].x,results.pose_landmarks.landmark[25].y, results.pose_landmarks.landmark[23].z])
            leftLeg[27] = np.array([results.pose_landmarks.landmark[27].x,results.pose_landmarks.landmark[27].y, results.pose_landmarks.landmark[23].z])
                
            rightLeg[24] = np.array([results.pose_landmarks.landmark[24].x,results.pose_landmarks.landmark[24].y, results.pose_landmarks.landmark[23].z])
            rightLeg[26] = np.array([results.pose_landmarks.landmark[26].x,results.pose_landmarks.landmark[26].y, results.pose_landmarks.landmark[23].z])
            rightLeg[28] = np.array([results.pose_landmarks.landmark[28].x,results.pose_landmarks.landmark[28].y, results.pose_landmarks.landmark[23].z])


            
            # cv2.circle(frame, (cx,cy),3,(255,0,0),cv2.FILLED)
            
            # print(
            # results.pose_landmarks[15],
            # results.pose_landmarks[17],
            # results.pose_landmarks[19],
            # )
            # print(results.pose_landmarks)
    # display output

    print(leftLeg, rightLeg)
    if (23 in leftLeg):
        a = leftLeg[23]
    if (25 in leftLeg):
    
        b = leftLeg[25]
    if (27 in leftLeg):
    
        c = leftLeg[27]

    if (a!=None and b!=None and c!=None):
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
    
        print (np.degrees(angle))

    cv2.imshow("gender detection", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()