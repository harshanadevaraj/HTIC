import cv2
import numpy as np
import math
image = cv2.imread("/Users/harshanad/Downloads/monarch-butterfly-male.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)
def gkernel(size, sigma):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    total = 0.0
    for x in range(size):
        for y in range(size):
            x_dist = x - center
            y_dist = y - center
            val = (1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x_dist ** 2 + y_dist ** 2) / (2 * sigma ** 2))
            kernel[x, y] = val
            total += val
    kernel /= total  
    return kernel
def fil(image, kernel):
    ksize = kernel.shape[0]
    pad = ksize // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode='edge')
    end = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
         
            region = padded[i:i+ksize, j:j+ksize]
            end[i, j] = np.sum(region * kernel)
    return np.clip(end, 0, 255).astype(np.uint8)
ker = gkernel(11, 6)         
result = fil(gray, ker)
cv2.imshow('Result', result)   
cv2.waitKey(0)
cv2.destroyAllWindows()