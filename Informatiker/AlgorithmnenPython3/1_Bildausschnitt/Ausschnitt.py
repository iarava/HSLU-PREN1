import cv2

PathImg = 'D:\Schule\PREN\Bilder\\'
InputImg = 'Zwei.jpg'

img = cv2.imread(PathImg+InputImg)
crop_img = img[500:1500, 500:1500]
#crop_img = cv2.resize(crop_img,(500,500))
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)