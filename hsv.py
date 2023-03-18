import cv2 as cv
import numpy as np
import imutils
cap = cv.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([107, 123 ,102])
    upper_blue = np.array([180,255, 255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)

    a = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    a = imutils.grab_contours(a)
    if (len(a) > 0): 
        c = max(a, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        except:
            center = 0
		# préciser le rayon minimal pour la détection (en pixels)
        if radius > 10:
		# dessiner le cercle et mettre à jour la liste des dernières coordonnées (pour le trail)
            cv.circle(mask, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv.circle(mask, center, 5, (0, 0, 255), -1)			
			#conversion en centimètres
            x=((x/1000)*64)+1
            y=((y/1000)*64)-0.4
            x=round(x,1)
            y=round(y,1)
            cv.putText(mask,"x,y: "+str(x)+","+str(y),(20,20),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
    cv.drawContours(mask, a, 0, (255,255,255),2)
    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(5) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()