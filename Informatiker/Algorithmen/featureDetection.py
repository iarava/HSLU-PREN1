import cv2 as cv
from matplotlib import pyplot as plt

Path = 'D:\Schule\PREN\Bilder'
ImageRef = '\Sechs.png'
InputImage = '\SechsausZahlen.png'

ref = cv.imread(Path+ImageRef,0)          # queryImage
scene = cv.imread(Path+InputImage,0) # trainImage

method = 'ORB'  # 'SIFT'
lowe_ratio = 0.89

if method   == 'ORB':
    finder = cv.ORB_create(nfeatures=2000)
elif method == 'SIFT':
    finder = cv.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = finder.detectAndCompute(ref,None)
kp2, des2 = finder.detectAndCompute(scene,None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []

for m,n in matches:
    if m.distance < lowe_ratio*n.distance:
        good.append([m])

print('using %s with lowe_ratio %.2f' % (method, lowe_ratio))
print('there are %d good matches' % (len(good)))

img3 = cv.drawMatchesKnn(ref,kp1,scene,kp2,good, None, flags=2)
plt.imshow(img3)
plt.show()