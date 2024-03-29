import cv2
import numpy as np

dataset = cv2.CascadeClassifier('data.xml')

cap = cv2.VideoCapture(0)
facedata = []
while True:
    boolean, image = cap.read()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = dataset.detectMultiScale(gray,1.28)
    for x,y,w,h in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),5)
        # face = image[y:y + h, x:x + w, :]

        face = gray[y:y+h,x:x+w]


        face = cv2.resize(face, (50,50))
        if len(facedata) < 200:
            facedata.append(face)
            print(len(facedata))

    # cv2.imshow('result',image)
    cv2.imshow('result', image)
    if cv2.waitKey(10) == 27 or len(facedata) >= 200:
        break

facedata = np.asarray(facedata)
# face_1 = Ashish
# face_2 = Neelam
# face_3 = Ashok
# face_4 = Vipul
# face_5 = HArsh
# face_6 = Kushal
# face_7 = Vishal
# face_8 = Janak
# face_9 = Vushnu
# face_10 = Mata g
# face_11 = Jyoti MAta g
np.save('face_1.npy',facedata)

cap.release()
cv2.destroyAllWindows()