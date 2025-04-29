# Carregar imagem
import cv2
import numpy as np
#imagem 1
img1 = cv2.imread('casa.jpg', cv2.IMREAD_GRAYSCALE)

harris = cv2.cornerHarris(img1, blockSize=3, ksize=3, k=0.04)
harris = cv2.dilate(harris, None)

img1[harris > 0.01 * harris.max()] = [255]

#imagem 2
img2 = cv2.imread('casa2.jpg', cv2.IMREAD_GRAYSCALE)

harris = cv2.cornerHarris(img2, blockSize=5, ksize=7, k=0.02)
harris = cv2.dilate(harris, None)

img2[harris > 0.01 * harris.max()] = [255]

#imagem 3
img3 = cv2.imread('sala.jpg', cv2.IMREAD_GRAYSCALE)

harris = cv2.cornerHarris(img3, blockSize=8, ksize=1, k=0.04)
harris = cv2.dilate(harris, None)

img3[harris > 0.01 * harris.max()] = [255]

cv2.imshow("Harris Corner ex1", img1)
cv2.imshow("Harris Corner ex2", img2)
cv2.imshow("Harris Corner ex3", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Shi-Tomasi

import cv2
import numpy as np

img1 = cv2.imread('sala.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('sala.jpg', cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread('sala.jpg', cv2.IMREAD_GRAYSCALE)

#Imagem 1
cantos = cv2.goodFeaturesToTrack(img1, maxCorners=100, qualityLevel=0.04, minDistance=10)
cantos = np.int64(cantos)


for canto in cantos:
    x, y = canto.ravel()
    cv2.circle(img1, (x, y), 3, 255, -1)


#Imagem 2
cantos = cv2.goodFeaturesToTrack(img2, maxCorners=100, qualityLevel=0.04, minDistance=10)
cantos = np.int64(cantos)


for canto in cantos:
    x, y = canto.ravel()
    cv2.circle(img3, (x, y), 3, 255, -1)

cantos = cv2.goodFeaturesToTrack(img2, maxCorners=100, qualityLevel=0.04, minDistance=10)
cantos = np.int64(cantos)

#Imagem 3
for canto in cantos:
    x, y = canto.ravel()
    cv2.circle(img3, (x, y), 3, 255, -1)

cantos = cv2.goodFeaturesToTrack(img3, maxCorners=100, qualityLevel=0.04, minDistance=10)
cantos = np.int64(cantos)


for canto in cantos:
    x, y = canto.ravel()
    cv2.circle(img3, (x, y), 3, 255, -1)

cv2.imshow('Shi-Tomasi 1 ', img1)
cv2.imshow('Shi-Tomasi 2 ', img2)
cv2.imshow('Shi-Tomasi 3 ', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# SIFT

import cv2
import numpy as np


img1 = cv2.imread('sala.jpg', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()


keypoints, descriptors = sift.detectAndCompute(img1, None)


imagem_sift = cv2.drawKeypoints(img1, keypoints, None)


cv2.imshow('SIFT', imagem_sift)
cv2.waitKey(0)
cv2.destroyAllWindows()








