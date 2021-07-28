import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
#you can use any trained model as per convenience, our best accuracy model was mobilenet
model = load_model('mobilenetTrainedFER.h5')
labels=['surprise', 'fear', 'angry', 'neutral', 'sad', 'disgust', 'happy']

def fr(img):
    #cv.imshow('Group of 5 people', img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cv.imshow('Gray People', gray)

    haar_cascade = cv.CascadeClassifier('haar_cascade.xml')

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    #print(f'Number of faces found = {len(faces_rect)}')
    flag=False

    for (x,y,w,h) in faces_rect:
        imgcrop=img[y:y+w,x:x+h]
        flag=True
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
        cv.putText(img,"fer",(x,y+w+18),cv.FONT_HERSHEY_DUPLEX,1,(0,0,255),1)
             
    if flag==True:
        imgfinal=cv.resize(imgcrop,(224,224))
        cv.imshow('cropedim',imgfinal)
        im=np.expand_dims(imgfinal, axis=0)
        pri=model.predict(im)
        dic={}
        for i in range(len(pri[0])):
            dic[labels[i]]='%.2f' % (pri[0][i]*100)
        print(dic)
        
    cv.imshow('Detected Faces', img)


cv.waitKey(0)

web=cv.VideoCapture(0) #for webcam
web.set(3,800)# 3 is id for length
web.set(4,600)# 4 is id for breadth
web.set(10,60)#10 is id for brightness
while True:
    success,img=web.read()
    fr(img)
    #cv.imshow("video",img)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
web.release()
cv.destroyAllWindows()
