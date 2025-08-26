#GROUP # DYPCET TY A
import cv2
import math
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained Keras model
model = load_model("Model/keras_model.h5")

# Load gesture labels
labels = open("Model/labels.txt").read().splitlines()

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        
        aspectRatio = h / w
        
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        # Preprocess the image and make prediction
        imgWhite = cv2.resize(imgWhite, (224, 224))  
        imgWhite = imgWhite / 255.0  
        imgWhite = np.expand_dims(imgWhite, axis=0)  

        predictions = model.predict(imgWhite)
        classIndex = np.argmax(predictions)  
        confidence = predictions[0][classIndex]

       
        cv2.putText(img, f'{labels[classIndex]}: {confidence*100:.2f}%', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
