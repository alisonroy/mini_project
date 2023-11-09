import numpy as np
import cv2
import tensorflow as tf
import keras

detector = cv2.CascadeClassifier('face.xml')
cap = cv2.VideoCapture(0)
count = 0
while(True):
    ret, img = cap.read()
    faces = detector.detectMultiScale(img, 1.3, 5)
    if(len(faces) != 0 and type(faces) != None):
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]  # slice the face from the image
            test_image = np.expand_dims(face, axis=0)
            cv2.imwrite('./dataset/alison/'+str(count)+'_img.png', face)
            count += 1
    cv2.imshow('frame', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
