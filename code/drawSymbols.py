
from predictClass import * 
from keyboard import *
from doAction import *

from collections import deque
import numpy as np
import imutils
import cv2
import urllib  as ul


greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

blueLower = (110,50,50)
blueUpper = (130,255,255) 

pts = deque(maxlen=100)


#url = "http://192.168.1.9:8080/shot.jpg"
#imgResp = ul.request.urlopen(url)
#imgNP = np.array(bytearray(imgResp.read()),dtype=np.uint8)
#output = cv2.imdecode(imgNP,-1)
#ooutput= cv2.flip( output, 1 )


cap = cv2.VideoCapture(0)
_,output = cap.read()

output= imutils.resize(output, width=400)
output= cv2.GaussianBlur(output, (11, 11), 0)
output[::]=[0]

while True:
    
    clear = False
#    (grabbed, frame) = camera.read()
    #imgResp = ul.request.urlopen(url)
    #imgNP = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    #frame = cv2.imdecode(imgNP,-1)
#    frame = cv2.flip( frame , 1 )
    
    _,frame = cap.read()
    
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=400)
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
#    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.inRange(hsv, blueLower, blueUpper)
    
    temp_mask = mask 
    
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            pass
#            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # update the points queue
    if center != None:
        pts.appendleft(center)

    # loop over the set of tracked points
#    print(radius,len(cnts))
    if radius > 15:
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue
    
            # otherwise, compute the thickness of the line and
            # draw the connecting lines
    #        thickness = int(np.sqrt(500/ float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 5)
            cv2.line(output, pts[i - 1], pts[i], (0, 0, 255), 5)
    #        print(radius)
    
    else:    
#        print(len(pts))
        if len(pts)>5 :
            cv2.imwrite(r'../TESTING/saved.png',output)
            action = predictClass()
            #action  = action[2] > 0.5 ? action[0] : -1; 
            if action[2] > 0.5:
                action = action[0]
            else:
                action = -1
            actionStatus = act(action)
            clear = True
            output[::]=[0]
    
    if clear :
        pts.clear()

    if actionStatus == True:
        print("Action Done ")
    elif actionStatus == False:
        print("Action Failed ")
    elif actionStatus == None:
        print("Action class Error ")

    # show the frame to our screen
    cv2.imshow("output",output)
#    cv2.imshow("temp mask ",temp_mask)
    cv2.imshow("Frame", frame)
    #cv2.imshow("mask", mask)
    
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break
    radius = 0 

# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()