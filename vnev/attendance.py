
from datetime import datetime
from pydoc import classname
from unicodedata import name
import cv2
from cv2 import cvtColor
import numpy as np
import face_recognition 

import sys


# imgRdj=face_recognition.load_image_file('C:\python opencv project\ImagesBasic\Robert-Downey.jpg') 
# imgRdj=cv2.cvtColor(imgRdj,cv2.COLOR_BGR2RGB) 

# imgTest=face_recognition.load_image_file('C:\python opencv project\ImagesBasic\download.jpg') 
# imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

#reading images through file

import os 

path='C:\python opencv project\ImagesAttendance' 

#Images list to store images
Images=[]

#ClassNames to store only name attendee
ClassNames=[]

#mylist will contain all name of attendee
mylist=os.listdir(path) 

# print(mylist)
for img in mylist:
    curimg=cv2.imread(f'{path}/{img}')
    Images.append(curimg)
    ClassNames.append(os.path.splitext(img)[0])
# print(ClassNames)  


#function to encode images
def encodeImage(Images):
    encodeList=[] 
    for img in Images:
        cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0] 
        encodeList.append(encode)
    return encodeList
  


#function to mark attendance of new candidate
def markAttendance(name):
    with open('attendanceList.csv','r+') as f:
        myDatalist=f.readlines()
        nameList=[]
        for line in myDatalist:
            entry=line.split(',')
            nameList.append(entry[0]) 
        if name not in nameList:
            now=datetime.now()
            date_time_string=now.strftime('%H: %M :%S')
            f.writelines(f'\n{name},{date_time_string}')
        else:
            print("Your attendance is already marked "+ name)        




#storing encoded images of attendee  
KnownAttendee=encodeImage(Images)
print("Encoding completed!")      


#on the webcam to capture face 
cap=cv2.VideoCapture(0) 


#while camera is on 
while True:
    success,img=cap.read() 
    imgSmall=cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall=cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)
    
    facesCurFrame=face_recognition.face_locations(imgSmall)
    
    encodesCurFrame=face_recognition.face_encodings(imgSmall,facesCurFrame)
    
    for encodeface,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches=face_recognition.compare_faces(KnownAttendee,encodeface)
        faceDis=face_recognition.face_distance(KnownAttendee,encodeface)
        # print(faceDis)
        matchIndex=np.argmin(faceDis)
        
        if matches[matchIndex]:
            name=ClassNames[matchIndex].upper()
            
            # print(name)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=4*y1,4*x2,4*y2,4*x1
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),2)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            markAttendance(name)
  

            
           
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
    
        



