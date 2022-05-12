


import cv2
import numpy as np 
import face_recognition   

imgRdj=face_recognition.load_image_file('C:\python opencv project\ImagesBasic\Robert-Downey.jpg') 
imgRdj=cv2.cvtColor(imgRdj,cv2.COLOR_BGR2RGB) 

imgTest=face_recognition.load_image_file('C:\python opencv project\ImagesBasic\download.jpg') 
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB) 

faceLoc=face_recognition.face_locations(imgRdj)[0] 
encodeRdj=face_recognition.face_encodings(imgRdj)[0] 

cv2.rectangle(imgRdj,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) 

faceLocTest=face_recognition.face_locations(imgTest)[0] 
encodeTest=face_recognition.face_encodings(imgTest)[0] 
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)


results=face_recognition.compare_faces([encodeRdj],encodeTest)
faceDis=face_recognition.face_distance([encodeRdj],encodeTest)

print(results,faceDis[0])
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,255),2)
cv2.imshow('Robert-Downey',imgRdj)
cv2.imshow('Robert-Downey-Test',imgTest) 

cv2.waitKey(0)