# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm
import cv2

Path = 'D:\Schule\PREN\Bilder\\'
InputImage = 'ZahlWeissAufSchwarz.jpg'

digits = datasets.load_digits()
data = digits.data
target = digits.target
images = digits.images
print('Data: ', data)
print('Target: ', target)
print('Images', images)

img = cv2.imread(Path+InputImage,cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(64,64))

classifier = svm.SVC()
classifier.fit(np.reshape(images,(len(digits.images),-1)),target)

prediction = classifier.predict(img)
print(prediction)