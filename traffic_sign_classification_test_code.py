import numpy as np
import cv2
from keras.models import load_model

# Load trained model (assumes .h5 format)
model_path = r"Desktop\model_trained.h5"
model = load_model(model_path)

# Constants
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

# Setup camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

# Preprocessing functions
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

# Class name lookup
def getClassName(classNo):
    classes = ['Speed Limit 20 km/h', 'Speed Limit 30 km/h', ..., 'End of no passing by vehicles over 3.5 metric tons']
    return classes[classNo] if 0 <= classNo < len(classes) else "Unknown"

# Live prediction loop
while True:
    success, imgOriginal = cap.read()
    if not success:
        break

    img = cv2.resize(imgOriginal, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    predictions = model.predict(img)
    classIndex = int(np.argmax(predictions))
    probabilityValue = np.max(predictions)

    if probabilityValue > threshold:
        cv2.putText(imgOriginal, f"{classIndex} {getClassName(classIndex)}", (20, 35), font, 0.75, (0, 0, 255), 2)
        cv2.putText(imgOriginal, f"{round(probabilityValue * 100, 2)}%", (20, 75), font, 0.75, (0, 0, 255), 2)

    cv2.imshow("Processed Image", img.reshape(32, 32))
    cv2.imshow("Result", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
