import cv2
import numpy as np

vid = cv2.VideoCapture("http://192.168.201.91:4747/video")

while(True):
    ret, f = vid.read()
    frame = f.copy()[12: ,]
    gray = gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bi = cv2.bilateralFilter(gray, 5, 75, 75)
    cv2.imshow('bi',bi)
    
    dst = cv2.cornerHarris(bi, 2, 3, 0.045)
    mask = np.zeros_like(gray)
    mask[dst>0.01*dst.max()] = 255
    cv2.imshow('mask', mask)

    # cv2.imshow('dst',dst)
    cv2.imshow("orig", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break