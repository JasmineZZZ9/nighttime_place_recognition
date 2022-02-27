
import cv2
import numpy as np

img = cv2.imread('midterm/nighttime place recognition dataset/train/00001323/20151102_004020.jpg')
img = cv2.resize(img, (640, 480))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# 输入图像必须是float32， 最后一个参数[0.04,0.06]
dst = cv2.cornerHarris(gray, 3, 5, 0.04)
#cv2.imshow('dst', dst)
#dst = cv2.dilate(dst, None)

a = dst > 0.01 * dst.max()
img[dst > 0.01 * dst.max()] = [0, 0, 255]
#cv2.imshow('img', img)
#cv2.imshow('dst2', dst)
print(type(dst))
print(dst)
print(type(a))
print(a)

num = 0

[rows, cols] = a.shape
print(rows, cols)
for i in range(rows):
    for j in range(cols):
        if a[i][j] == True:
            num += 1


print('num=', num)
cv2.waitKey(0)
cv2.destroyAllWindows()

 
