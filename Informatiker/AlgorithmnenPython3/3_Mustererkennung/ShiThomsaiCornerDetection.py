import numpy as np
import cv2
from matplotlib import pyplot as plt

PathImg = 'D:\Schule\PREN\Bilder\\'
InputImg = 'ZahlenSchwarzAufWeiss.jpg'

img = cv2.imread(PathImg+InputImg)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Parameters for Shi-Tomasi algorithm
maxCorners = 20
qualityLevel = 0.01
minDistance = 20
blockSize = 3
gradientSize = 3
useHarrisDetector = False
k = 0.04

corners = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance, None, blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img),plt.show()