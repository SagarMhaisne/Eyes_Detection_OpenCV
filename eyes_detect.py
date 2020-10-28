# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 15:19:42 2020

@author: SM51998
"""

import cv2
import numpy as np

#cascade file for eye stored in local directory
cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#reading the image
img = cv2.imread('2.jpg')

#resizing the image to 500x500
imp = cv2.resize(img,(500,500))

copy = img.copy()

# converting image to gray
gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)

#for detecting the eyes in the image
eyes = cascade.detectMultiScale(gray,1.3,5)

#dimensions for rectangle
for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(copy,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    #image, dim from, dim to, color of rect, width of rect

cv2.imshow('Original',img)
cv2.imshow('Eyes detected', copy)
stack = np.hstack([img,copy])  #horizontal stacking of images
cv2.imshow('Output',stack)
#cv2.waitKey(0) # infifnite delay