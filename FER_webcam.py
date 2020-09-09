import numpy as np
import cv2
import sys
from keras.models import model_from_json

class FER_Model(object):

    def __init__(self, json_model, weight_model):
        with open(json_model, "r") as json_file:
            loaded_json_model = json_file.read()
            self.model = model_from_json(loaded_json_model)

        self.model.load_weights(weight_model)
        self.model.summary()


    def predict_emotion(self, img):
        emotions = ["Happy", "Sad", "Neutral"]
#         emotions = ["Angry", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        return emotions[np.argmax(self.model.predict(img))]

model = FER_Model("models/model_3class.json", "models/model_3class.h5")


cascPath = 'haarcascade_frontalface_default.xml'
font = cv2.FONT_HERSHEY_SIMPLEX
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        roi = cv2.resize(face, (48,48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        cv2.putText(frame, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
