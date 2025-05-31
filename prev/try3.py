import cv2 
import numpy as np

vid = cv2.VideoCapture("http://192.168.201.246:4747/video")

while True:
    ret, f = vid.read()
    frame = f.copy()[12: ,]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    cv2.imshow("orig", gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break