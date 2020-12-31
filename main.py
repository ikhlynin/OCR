import numpy as np
import cv2
import pickle

##### Global settings #####
width = 800
height = 800
threshold = 0.65
camera = 0


##### set up webcam #####
cap = cv2.VideoCapture(camera)
cap.set(3, width)
cap.set(4, height)


##### load da model
pickle_in = open("model_train.p","rb")
model = pickle.load(pickle_in)


##### irl preprocessing #####
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

##### start #####
while True:
    success, suorce_img = cap.read()
    img = np.asarray(suorce_img)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processsed Image", img)
    img = img.reshape(1, 32, 32, 1)
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    probability= np.amax(predictions)
    print(classIndex, probability())

    if probability> threshold:
        cv2.putText(suorce_img,
                    str(classIndex) + "   "+str(probability),
                    (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 1)

    cv2.imshow("Original Image",suorce_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break