import cv2
import numpy as np
import cv2.aruco as aruco
'''
File Name:perspective
Funtions:mainarea
Global Variable:
                    
'''
'''
Function Name:mainarea
Input:img_rgb(image from camera)
Output:back box enclosed area in image or returns the same image
Logic:contours of specific area in selected to fet the largest black box in area
Example Call:
'''
calibration = np.load("calibrated.npz", allow_pickle=False)
mtx=calibration["mtx"]
dist=calibration["dist"]

def mainarea(img_rgb):
    
    
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
    	contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        
        cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        
        x=0 
        y=0 
        w=0 
        h=0
    if contours:
       # for contour in contours:
            
            #approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
            #area = cv2.contourArea(contour) #calculate the area of contour detected
        contour= max(contours, key = cv2.contourArea)
        
            #if (area >100000)and area<250000 : # true if area is between range may vary with the height of camera set each time when camera height is changed
                

        


                
        x,y,w,h = cv2.boundingRect(contour) #gives x,y,w,h of rectangle around contour
                
        cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,0,255),2) #draw bounding rectangle around rectangle 
                #print x,y,w,h
        if w>0 and h>0:
            
                crop_img = img_rgb[y:y+h, x:x+w] #crop black rectangel from the img_rgb
                pts1 = np.float32([[x,y],[x+w,y],[x,y+h],[x+w,y+h]]) # rectangle for perspective transform
                pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]]) #canvas size for perspective transform
                M = cv2.getPerspectiveTransform(pts1,pts2) #perspective transform of the img_rgb


                arena = cv2.warpPerspective(crop_img,M,(w,h)) #wrap perspectie
            #cv2.imshow('Arena',arena)
     
        
    
                h, w = arena.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
	# undistort
                dst = cv2.undistort(arena, mtx, dist, None, newcameramtx)
     #crop the image
                x, y, w, h = roi
                arena = dst[y:y+h, x:x+w]
                cv2.imshow('cropped',arena)
                return arena #returns image of black box arena
        #    else:
        #        return img_rgb #returns image of img_rgb as it is

'''
to make the script stand alone and check persperctive transform of black box arena
decomment the below code change the comport if required
'''

cap=cv2.VideoCapture(0)
while(1):
    
    ret,img_rgb=cap.read()
    #img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
	
    if ret:
        mainarea(img_rgb)
    gray=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
	#cv2.imshow('Gray',img_gray)
    rows,cols,ch = img_rgb.shape
    cv2.imshow('image',img_rgb)
    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):

        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
        #(rvec-tvec).any() # get rid of that nasty numpy value array error

        for i in range(0, ids.size):
            # draw axis for the aruco markers
            aruco.drawAxis	 (img_rgb, mtx, dist, rvec[i], tvec[i], 0.1)

        # draw a square around the markers
        aruco.drawDetectedMarkers(img_rgb, corners)


        # code to show ids of the marker found
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0])+', '

        cv2.putText(img_rgb, "Id: " + strg, (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)


    else:
        # code to show 'No Ids' when no markers are found
        cv2.putText(img_rgb, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

    
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        
        break
    
