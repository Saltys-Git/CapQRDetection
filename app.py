import cv2
import numpy as np
import math
from glob import glob





def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang if ang < 0 else ang

kernel = np.ones((5,5),np.uint8)

path='temp/*.jpg'

for fsl, file in enumerate(glob(path)):

    #file='temp/20211126_165836.jpg'

    #print(file)
    img=cv2.imread(file)
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([0,0,168])
    upper_blue = np.array([172,111,255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(img, img, mask= mask)

    erosion = cv2.erode(res,kernel,iterations =1)

    gray= cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)

    opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

    contours0, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnumber = 0



    for sl, cnt in enumerate(contours0):
        x, y, w, h = cv2.boundingRect(cnt)
        #cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        #print(sl)
        crop_img=img[y:y+h, x:x+w]

        edges = cv2.Canny(crop_img, 40, 300)
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 1000,
                                   param1=60, param2=30, minRadius=0, maxRadius=0)

        try:
            for i in circles[0, :]:
                cnumber +=1
                cx = int(i[0])
                cy = int(i[1])
                r = int(i[2])
                #cv2.circle(img, (y+cx, x+cy), r, (0,0,255), -1)
                #print(cnumber)


                angle = int(getAngle((x, y), (x + w, y + h), (x+h, x)))

                #angle=getAngle((200,200), (400,400), (400,200))
                if (angle >= 40 and angle <= 50) or (angle >= 220 and angle <= 230):
                    img[y:y + h, x:x + w][:, :, 2] = 255

                #cv2.putText(img, f'{int(angle)}', (x-20, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 5)
                #print(angle, sl)

    #cv2.putText(blank, 'A', (300,200), cv2.FONT_HERSHEY_COMPLEX, 1, (255), 2)

                #print(img[y:y + h, x:x + w][:,:,:][:,:,3].shape)
                #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                #cv2.imwrite(f'count/output{sl}.jpg', crop_img)
        except:
            #print("i am not able to detecti circle")
            pass

    #rimages=cv2.resize(img, (1800,1000))
    cv2.imwrite(f'out/output {fsl} .jpg', img)
    #print(fsl)
    #cv2.imshow('output', rimages)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    
    
    
