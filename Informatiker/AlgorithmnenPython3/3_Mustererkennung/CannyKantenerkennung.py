import cv2

PathImg = 'D:\Schule\PREN\Bilder\\'
InputImg = 'ZahlWeissAufSchwarz.jpg'

img = cv2.imread(PathImg+InputImg)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img,(500,500))
cv2.imshow("Startbild", img)
img = cv2.Canny(img, 100, 200)
cv2.imshow("result", img)
cv2.waitKey(0)