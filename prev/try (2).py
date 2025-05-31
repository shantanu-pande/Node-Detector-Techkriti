import cv2
import numpy as np


def getContours(img, imgContours):
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours, contours, -1, (255,0,255), 7)


def null(arg):
    # print(arg)
    pass
     

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver



vid = cv2.VideoCapture("http://192.168.0.31:4747/video")
cv2.namedWindow("para")
cv2.resizeWindow("para", 640, 450)
cv2.createTrackbar("thresh1", "para", 190,255, null)
cv2.createTrackbar("thresh2", "para", 95,255, null)

while(True):
    ret, f = vid.read()
    img = f.copy()[12: ,]

    imgCopy = img.copy()
    imgContours = img.copy()

  
    imgBlur = cv2.GaussianBlur(img, (7,7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    th1 = cv2.getTrackbarPos("thresh1", "para")
    th2 = cv2.getTrackbarPos("thresh2", "para")
    imgCanny = cv2.Canny(imgBlur, th1, th2)

    kernal = np.ones((5,5))
    imgDil = cv2.dilate(imgCanny, kernal, iterations=1)
    getContours(imgDil, imgContours)

    imgStack = stackImages(0.7, ([img, imgGray, imgCanny],
                                [imgDil, imgContours, imgDil]))
    cv2.imshow("image", imgStack)

    # cv2.imshow("orig", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break