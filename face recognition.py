import cv2


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img = cv2.imread('DSC_0770.jpg')#reading image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#converts to grayscale

img3 = cv2.resize(img,(1000,400))


faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors =9)
for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),5)
    #w and h are height and width,(0,0,255)-(B,G,R),5 - line thickness
    
cv2.imshow('FACE RECOGNITION',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()


img3 = cv2.resize(img,(1000,400))
cv2.imshow('CUSTOM DIMENSIONS',img3)
