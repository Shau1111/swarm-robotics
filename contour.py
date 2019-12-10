import cv2
import numpy as np
while True:
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
if ret == True:
 
    # Display the resulting frame
    cv2.imshow("Frame",frame)
 
    blur = cv2.GaussianBlur(frame,(5,5),0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([38,86,0])
    upper_blue = np.array([121,255,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for contour in contours:
    cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        camera.release()
        cv2.destroyAllWindows()
        
        break
