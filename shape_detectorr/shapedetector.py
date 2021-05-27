import cv2

img = cv2.imread('images/shapes.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # if you have a color image just convert to gray
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 75, 200)
copyOfImage = img.copy()


def detectShapes(imgCanny):
    contours, hierachy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            cv2.drawContours(copyOfImage, contour, -1, (255, 0, 0), 2)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            approxLen = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            objectType = "Unknown"
            if approxLen == 3:
                objectType = "Triangle"
            elif approxLen == 4:
                aspRatio = w / float(h)
                if 0.95 < aspRatio < 1.05:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif approxLen == 5:
                objectType = "Pentagon"
            elif approxLen == 6:
                objectType = "Hexagon"
            elif approxLen == 8:
                objectType = "Octagon"
            elif approxLen > 8:
                objectType = "Circle"

            cv2.rectangle(copyOfImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(copyOfImage, objectType, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                        (0, 255, 0), 2)
            print(objectType)


detectShapes(imgCanny)
cv2.imshow("image", copyOfImage)
cv2.waitKey(0)
