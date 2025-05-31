import cv2
import numpy as np
import mapp
import time

vid = cv2.VideoCapture("http://192.168.77.54:4747/video")

def mapp(h):
    h = h.reshape((4,2))
    hnew = np.zeros((4,2),dtype = np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h,axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

while(True):
    ret, frame = vid.read()
    orig = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred=cv2.GaussianBlur(gray,(5,5),0)
    # blurred = cv2.bilateralFilter(gray, 5, 75, 75)

    edged=cv2.Canny(blurred,30,50)

    contours,hierarchy=cv2.findContours(edged,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)  #retrieve the contours as a list, with simple apprximation model
    contours=sorted(contours,key=cv2.contourArea,reverse=True)
    print(contours[0])
    for c in contours:
        p=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*p,True)

        if len(approx)==4:
            target=approx
            break
    approx=mapp(target) #find endpoints of the sheet

    pts=np.float32([[0,0],[800,0],[800,800],[0,800]])  #map to 800*800 target window

    op=cv2.getPerspectiveTransform(approx,pts)  #get the top or bird eye view effect
    dst=cv2.warpPerspective(orig,op,(800,800))


    cv2.imshow('f1', gray)
    cv2.imshow('f2', edged)
    cv2.imshow('frame', dst)
    time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


vid.release()
cv2.destroyAllWindows()