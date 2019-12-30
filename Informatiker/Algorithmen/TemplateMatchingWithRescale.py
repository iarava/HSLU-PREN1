# USAGE
# python match.py --template cod_logo.png --images images

# import the necessary packages
import numpy as np
import imutils
import cv2
import time
import matplotlib.pyplot as plt

Path = 'D:\Schule\PREN\Bilder'
ImageRef = '\SechsausZahlen.png'
InputImage = '\Zahlen2.png'


image = cv2.imread(Path+InputImage)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
found = None
found_edged = None
found_template = None

start_time = time.time()

image = cv2.imread(Path+InputImage)

# loop over the scales of the image
imagePath = Path+ImageRef
template = cv2.imread(imagePath)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
print 'Test img: ', imagePath
for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    # resize the image according to the scale, and keep track
    # of the ratio of the resizing
    resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
    r = gray.shape[1] / float(resized.shape[1])

    # if the resized image is smaller than the template, then break
    # from the loop
    if resized.shape[0] < tH or resized.shape[1] < tW:
        break

    # detect edges in the resized, grayscale image and apply template
    # matching to find the template in the image
    edged = cv2.Canny(resized, 50, 200)
    result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

    # if we have found a new maximum correlation value, then ipdate
    # the bookkeeping variable
    if found is None or maxVal > found[0]:
        found = (maxVal, maxLoc, r)
        found_edged = edged
        found_template = template

# unpack the bookkeeping varaible and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
if(found != None):
    (_, maxLoc, r) = found

    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
else:
    endX = 0
    endY = 0
    startX = 0
    startY = 0

total_time = time.time() - start_time
print 'Total time: ',total_time

# draw a bounding box around the detected result and display the image
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Image'), plt.xticks([]), plt.yticks([])

# cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
plt.subplot(122),plt.imshow(found_edged,cmap = 'gray')
plt.title('Edged Image'), plt.xticks([]), plt.yticks([])
plt.show()
