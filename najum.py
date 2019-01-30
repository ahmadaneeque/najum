import numpy as np
import cv2

# faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
img = cv2.imread('C:/packages/najum/data/abba.png')

# print(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print (gray)
cv2.imshow('img',gray)
cv2.waitKey(0)
# faces = face_cascade.detectMultiScale(gray, .1, 10)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
)
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     # eyes = eye_cascade.detectMultiScale(roi_gray)
#     # for (ex,ey,ew,eh) in eyes:
#     #     cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()