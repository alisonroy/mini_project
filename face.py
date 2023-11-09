import numpy as np
import cv2
import tensorflow as tf
import keras

detector = cv2.CascadeClassifier('face.xml')
classifier = keras.models.load_model(
    'model.h5', custom_objects=None, compile=True, options=None)
cap = cv2.VideoCapture(0)
count = 0
while(True):
    ret, img = cap.read()
    faces = detector.detectMultiScale(img, 1.3, 5)
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]  # slice the face from the image
        test_image = np.expand_dims(face, axis=0)
        try:
            result = classifier.predict(test_image, verbose=0)
            print('Prediction is: ', np.argmax(result))
        except ValueError:
            print("Unknown Person")
            # # print(training_set.class_indices)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # cv2.imshow('frame', img)
    # key = cv2.waitKey(1)
    # if key == ord('q'):
    #     break
# cv2.destroyAllWindows()
# cap.release()
