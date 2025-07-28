import cv2
import numpy as np
import math
img=cv2.imread("/Users/harshanad/Downloads/monarch-butterfly-male.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
x=np.array(
   [[-1,0,1],
    [-2,0,2],
    [-1,0,1]])
y=np.array(
    [[-1,-2,-1],
    [0,0,0],
    [1,2,1]])
def sobelfilte(img,x,y):
    padding=np.pad(img,pad_width=1,mode='constant',constant_values=0) 
    output=np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region=padding[i:i+3,j:j+3]
            dotx=np.sum(region*x)
            doty=np.sum(region*y)
            magnitude=np.sqrt(dotx**2+doty**2)
            output[i,j]=magnitude
    return output                                                       
res=sobelfilte(gray,x,y)
cv2.imshow('sobel',res)
cv2.imshow('original',img)
cv2.imshow('gray',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
