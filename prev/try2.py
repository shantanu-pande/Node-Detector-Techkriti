import cv2
import numpy as np

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area


def reorder(myPoints):
 
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = myPoints.sum(1)
 
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] =myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] =myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
 
    return myPointsNew

vid = cv2.VideoCapture("http://192.168.201.91:4747/video")
while(True):
    ret, f = vid.read()
    frame = f.copy()[12: ,]
    # print(str(frame.shape))
    heightImg = 468
    widthImg  = 640
    gray = gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(gray, (5, 5), 1)
    edged=cv2.Canny(imgBlur,30,50)

    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(edged, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    imgContours = frame.copy()
    imgBigContour = frame.copy()

    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    biggest, maxArea = biggestContour(contours)
    print(biggest.size)
    if biggest.size != 0:
        biggest=reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # DRAW THE BIGGEST CONTOUR
        # imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
 
        #REMOVE 20 PIXELS FORM EACH SIDE
        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv2.resize(imgWarpColored,(widthImg,heightImg))
 
        # APPLY ADAPTIVE THRESHOLD
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)
    # corners = cv2.goodFeaturesToTrack(edged, 10, 0.04, 50)
    # for corner in corners:
         
    #     # cv2.circle(frame, tuple(reversed(corner)), 3, (0, 0, 255), -1)
    #     cv2.circle(frame, (int(corner[0][0]),int(corner[0][1])), 6, (0, 0, 255), -1)
    # print(f"corner : {corners}\n")
    cv2.imshow("orig", imgContours)
    cv2.imshow("f", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break