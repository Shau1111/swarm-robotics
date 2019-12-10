import cv2
import numpy as np

cap=cv2.VideoCapture(1)
cap.set(3,1920)
cap.set(4,1080)
ret,frame=cap.read()
r = cv2.selectROI(frame)
while True:
	ret,frame=cap.read()
	cv2.imshow('color',frame)
	k=cv2.waitKey(1)
	if k==113:  
                break
	if k==115:
	        r = cv2.selectROI(frame)
 
print(r)
frameCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
cv2.imshow("Image", frameCrop)
k=cv2.waitKey(0)                
cap.release()
cv2.destroyAllWindows()
