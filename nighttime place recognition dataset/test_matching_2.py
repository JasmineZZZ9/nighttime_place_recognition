import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, 0.6) * 255.0, 0, 255)

daytime_img = cv.imread('./nighttime place recognition dataset/test/00021510/20151102_160120.jpg')
night_img = cv.imread('./nighttime place recognition dataset/test/00021510/20151102_060125.jpg')

# daytime_img = cv.imread('./nighttime place recognition dataset/test/00023966/20151120_114613.jpg')
# night_img = cv.imread('./nighttime place recognition dataset/test/00023966/20151120_191559.jpg')

# Denoise
night_img = cv.fastNlMeansDenoising(night_img)
daytime_img = cv.fastNlMeansDenoising(daytime_img)

# Gamma correction
night_img = cv.LUT(night_img, lookUpTable)
daytime_img = cv.LUT(daytime_img, lookUpTable)

# Harris corner
# night_gray = np.float32(cv.cvtColor(night_img, cv.COLOR_BGR2GRAY))
# daytime_gray = np.float32(cv.cvtColor(daytime_img, cv.COLOR_BGR2GRAY))
# night_dst = cv.cornerHarris(night_gray,2,3,0.04)
# night_dst = cv.dilate(night_dst,None)
# night_img[night_dst>0.01*night_dst.max()]=[0,0,255]
# daytime_dst = cv.cornerHarris(daytime_gray,2,3,0.04)
# daytime_dst = cv.dilate(daytime_dst,None)
# daytime_img[daytime_dst>0.01*daytime_dst.max()]=[0,0,255]
# cv.imwrite('./output/daytime_corner.jpg', daytime_img)
# cv.imwrite('./output/night_corner.jpg', night_img)

# Sharpen (Laplacian filter)
# kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
# daytime_imgLaplacian = cv.filter2D(daytime_img, cv.CV_32F, kernel)
# daytime_sharp = np.float32(daytime_img)
# daytime_img = daytime_sharp - daytime_imgLaplacian
# daytime_img = np.clip(daytime_img, 0, 255)
# daytime_img = daytime_img.astype('uint8')
# night_imgLaplacian = cv.filter2D(night_img, cv.CV_32F, kernel)
# night_sharp = np.float32(night_img)
# night_img = night_sharp - night_imgLaplacian
# night_img = np.clip(night_img, 0, 255)
# night_img = night_img.astype('uint8')

night_gray = cv.cvtColor(night_img, cv.COLOR_BGR2GRAY)
daytime_gray = cv.cvtColor(daytime_img, cv.COLOR_BGR2GRAY)

# Canny edge
# daytime_gray = cv.Canny(daytime_img,100,200)
# night_gray = cv.Canny(night_img,100,200)

# Find corners
kp_night = cv.goodFeaturesToTrack(night_gray, 5000, 0.01, 10)
kp_daytime = cv.goodFeaturesToTrack(daytime_gray, 5000, 0.01, 10)
l_night = []
for item in kp_night:
    l_night.append(item[0])
kp_night = np.array(l_night)
kp_night = cv.KeyPoint_convert(kp_night)
l_daytime = []
for item in kp_daytime:
    l_daytime.append(item[0])
kp_daytime = np.array(l_daytime)
kp_daytime = cv.KeyPoint_convert(kp_daytime)

# SIFT
sift = cv.SIFT_create()
# kp_night, des_night = sift.detectAndCompute(night_gray, None)
# kp_daytime, des_daytime = sift.detectAndCompute(daytime_gray, None)
# kp_night = sift.detect(night_gray)
# kp_daytime = sift.detect(daytime_gray)
kp_night, des_night = sift.compute(night_gray, kp_night)
kp_daytime, des_daytime = sift.compute(daytime_gray, kp_daytime)
night_sift_img = cv.drawKeypoints(night_gray, kp_night, night_img)
daytime_sift_img = cv.drawKeypoints(daytime_gray, kp_daytime, daytime_img)
cv.imwrite('./output/night_sift.jpg', night_sift_img)
cv.imwrite('./output/daytime_sift.jpg', daytime_sift_img)
bf = cv.BFMatcher()
matches = bf.knnMatch(des_night,des_daytime, k=2)
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
print(len(good))

match_img = cv.drawMatchesKnn(night_img, kp_night, daytime_img, kp_daytime,
                         good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(match_img)
plt.savefig('output/match_test12.jpg')