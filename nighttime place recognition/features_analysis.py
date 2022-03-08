import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_HEIGHT = 464
IMG_WIDTH = 640

list_features = np.zeros((5, 24), dtype=int)
pics_num = np.zeros((1,24), dtype=int)

# Harris
def countHarris(img, img_g):
    # cornerHarris(img, blockSize, ksize, k)
    dst = cv2.cornerHarris(img_g, 2, 3, 0.04)
    countHarris = 0
    a = dst > 0.01*dst.max()
    [rows, cols] = a.shape
    
    for i in range(rows):
        for j in range(cols):
            if a[i][j] == True:
                countHarris += 1
    
    return countHarris

# Shi-Tomasi
def countTomasi(img, img_g):
    # goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance...)
    dst = cv2.goodFeaturesToTrack(img_g, 0, 0.01, 10)
    dst = np.int0(dst)
    countTomasi = 0
    for i in dst:
        countTomasi += 1
    return countTomasi

# SIFT
def countSIFT(img, img_g):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img_g, None)
    countSIFT = 0
    for i in kp:
        countSIFT += 1
    return countSIFT

# SURF
def countSURF(img, img_g):
    surf = cv2.xfeatures2d.SURF_create()
    kp,des = surf.detectAndCompute(img_g, None)
    countSURF = 0
    for i in kp:
        countSURF += 1
    return countSURF

# ORB
def countORB(img, img_g):
    orb = cv2.ORB_create()
    kp,des = orb.detectAndCompute(img_g, None)
    countORB = 0
    for i in kp:
        countORB += 1
    return countORB

# number of features
def countFeatures(img, img_g, filename):
    if '_00' in filename:
        pics_num[0][0] += 1
        list_features[0][0] += countHarris(img, img_g)
        list_features[1][0] += countTomasi(img, img_g)
        list_features[2][0] += countSIFT(img, img_g)
        list_features[3][0] += countSURF(img, img_g)
        list_features[4][0] += countORB(img, img_g)
    if '_01' in filename:
        pics_num[0][1] += 1
        list_features[0][1] += countHarris(img, img_g)
        list_features[1][1] += countTomasi(img, img_g)
        list_features[2][1] += countSIFT(img, img_g)
        list_features[3][1] += countSURF(img, img_g)
        list_features[4][1] += countORB(img, img_g)
    if '_02' in filename:
        pics_num[0][2] += 1
        list_features[0][2] += countHarris(img, img_g)
        list_features[1][2] += countTomasi(img, img_g)
        list_features[2][2] += countSIFT(img, img_g)
        list_features[3][2] += countSURF(img, img_g)
        list_features[4][2] += countORB(img, img_g)
    if '_03' in filename:
        pics_num[0][3] += 1
        list_features[0][3] += countHarris(img, img_g)
        list_features[1][3] += countTomasi(img, img_g)
        list_features[2][3] += countSIFT(img, img_g)
        list_features[3][3] += countSURF(img, img_g)
        list_features[4][3] += countORB(img, img_g)
    if '_04' in filename:
        pics_num[0][4] += 1
        list_features[0][4] += countHarris(img, img_g)
        list_features[1][4] += countTomasi(img, img_g)
        list_features[2][4] += countSIFT(img, img_g)
        list_features[3][4] += countSURF(img, img_g)
        list_features[4][4] += countORB(img, img_g)
    if '_05' in filename:
        pics_num[0][5] += 1
        list_features[0][5] += countHarris(img, img_g)
        list_features[1][5] += countTomasi(img, img_g)
        list_features[2][5] += countSIFT(img, img_g)
        list_features[3][5] += countSURF(img, img_g)
        list_features[4][5] += countORB(img, img_g)
    if '_06' in filename:
        pics_num[0][6] += 1
        list_features[0][6] += countHarris(img, img_g)
        list_features[1][6] += countTomasi(img, img_g)
        list_features[2][6] += countSIFT(img, img_g)
        list_features[3][6] += countSURF(img, img_g)
        list_features[4][6] += countORB(img, img_g)
    if '_07' in filename:
        pics_num[0][7] += 1
        list_features[0][7] += countHarris(img, img_g)
        list_features[1][7] += countTomasi(img, img_g)
        list_features[2][7] += countSIFT(img, img_g)
        list_features[3][7] += countSURF(img, img_g)
        list_features[4][7] += countORB(img, img_g)
    if '_08' in filename:
        pics_num[0][8] += 1
        list_features[0][8] += countHarris(img, img_g)
        list_features[1][8] += countTomasi(img, img_g)
        list_features[2][8] += countSIFT(img, img_g)
        list_features[3][8] += countSURF(img, img_g)
        list_features[4][8] += countORB(img, img_g)
    if '_09' in filename:
        pics_num[0][9] += 1
        list_features[0][9] += countHarris(img, img_g)
        list_features[1][9] += countTomasi(img, img_g)
        list_features[2][9] += countSIFT(img, img_g)
        list_features[3][9] += countSURF(img, img_g)
        list_features[4][9] += countORB(img, img_g)
    if '_10' in filename:
        pics_num[0][10] += 1
        list_features[0][10] += countHarris(img, img_g)
        list_features[1][10] += countTomasi(img, img_g)
        list_features[2][10] += countSIFT(img, img_g)
        list_features[3][10] += countSURF(img, img_g)
        list_features[4][10] += countORB(img, img_g)
    if '_11' in filename:
        pics_num[0][11] += 1
        list_features[0][11] += countHarris(img, img_g)
        list_features[1][11] += countTomasi(img, img_g)
        list_features[2][11] += countSIFT(img, img_g)
        list_features[3][11] += countSURF(img, img_g)
        list_features[4][11] += countORB(img, img_g)
    if '_12' in filename:
        pics_num[0][12] += 1
        list_features[0][12] += countHarris(img, img_g)
        list_features[1][12] += countTomasi(img, img_g)
        list_features[2][12] += countSIFT(img, img_g)
        list_features[3][12] += countSURF(img, img_g)
        list_features[4][12] += countORB(img, img_g)
    if '_13' in filename:
        pics_num[0][13] += 1
        list_features[0][13] += countHarris(img, img_g)
        list_features[1][13] += countTomasi(img, img_g)
        list_features[2][13] += countSIFT(img, img_g)
        list_features[3][13] += countSURF(img, img_g)
        list_features[4][13] += countORB(img, img_g)
    if '_14' in filename:
        pics_num[0][14] += 1
        list_features[0][14] += countHarris(img, img_g)
        list_features[1][14] += countTomasi(img, img_g)
        list_features[2][14] += countSIFT(img, img_g)
        list_features[3][14] += countSURF(img, img_g)
        list_features[4][14] += countORB(img, img_g)
    if '_15' in filename:
        pics_num[0][15] += 1
        list_features[0][15] += countHarris(img, img_g)
        list_features[1][15] += countTomasi(img, img_g)
        list_features[2][15] += countSIFT(img, img_g)
        list_features[3][15] += countSURF(img, img_g)
        list_features[4][15] += countORB(img, img_g)
    if '_16' in filename:
        pics_num[0][16] += 1
        list_features[0][16] += countHarris(img, img_g)
        list_features[1][16] += countTomasi(img, img_g)
        list_features[2][16] += countSIFT(img, img_g)
        list_features[3][16] += countSURF(img, img_g)
        list_features[4][16] += countORB(img, img_g)
    if '_17' in filename:
        pics_num[0][17] += 1
        list_features[0][17] += countHarris(img, img_g)
        list_features[1][17] += countTomasi(img, img_g)
        list_features[2][17] += countSIFT(img, img_g)
        list_features[3][17] += countSURF(img, img_g)
        list_features[4][17] += countORB(img, img_g)
    if '_18' in filename:
        pics_num[0][18] += 1
        list_features[0][18] += countHarris(img, img_g)
        list_features[1][18] += countTomasi(img, img_g)
        list_features[2][18] += countSIFT(img, img_g)
        list_features[3][18] += countSURF(img, img_g)
        list_features[4][18] += countORB(img, img_g)
    if '_19' in filename:
        pics_num[0][19] += 1
        list_features[0][19] += countHarris(img, img_g)
        list_features[1][19] += countTomasi(img, img_g)
        list_features[2][19] += countSIFT(img, img_g)
        list_features[3][19] += countSURF(img, img_g)
        list_features[4][19] += countORB(img, img_g)
    if '_20' in filename:
        pics_num[0][20] += 1
        list_features[0][20] += countHarris(img, img_g)
        list_features[1][20] += countTomasi(img, img_g)
        list_features[2][20] += countSIFT(img, img_g)
        list_features[3][20] += countSURF(img, img_g)
        list_features[4][20] += countORB(img, img_g)
    if '_21' in filename:
        pics_num[0][21] += 1
        list_features[0][21] += countHarris(img, img_g)
        list_features[1][21] += countTomasi(img, img_g)
        list_features[2][21] += countSIFT(img, img_g)
        list_features[3][21] += countSURF(img, img_g)
        list_features[4][21] += countORB(img, img_g)
    if '_22' in filename:
        pics_num[0][22] += 1
        list_features[0][22] += countHarris(img, img_g)
        list_features[1][22] += countTomasi(img, img_g)
        list_features[2][22] += countSIFT(img, img_g)
        list_features[3][22] += countSURF(img, img_g)
        list_features[4][22] += countORB(img, img_g)
    if '_23' in filename:
        pics_num[0][23] += 1
        list_features[0][23] += countHarris(img, img_g)
        list_features[1][23] += countTomasi(img, img_g)
        list_features[2][23] += countSIFT(img, img_g)
        list_features[3][23] += countSURF(img, img_g)
        list_features[4][23] += countORB(img, img_g)


def read_path(file_pathname):
    for filename in os.listdir(file_pathname):
        if '.jpg' in filename:
            print(filename)
            img = cv2.imread(file_pathname+'/'+filename)
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
            # gray
            img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # number of features
            countFeatures(img, img_g, filename)
        else:
            continue
        
def read_path_all(path):
    for filename in os.listdir(path):
        if '.' not in filename:
            file_path = os.path.join(path, filename)
            read_path(file_path)
        else:
            continue

               
read_path_all("midterm/nighttime place recognition dataset/train")

average_feature = list_features/pics_num

print(average_feature[:3][:])

x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]#点的横坐标
k1 = average_feature[0][:]
k2 = average_feature[1][:]
k3 = average_feature[2][:]
k4 = average_feature[3][:]
k5 = average_feature[4][:]

plt.plot(x,k1,color = '#00ae9d',label="Harris Corner")
plt.plot(x,k2,color = '#f7acbc',label="Shi-Tomasi")
plt.plot(x,k3,color = '#7f7522',label="SIFT")
plt.plot(x,k4,color = '#f58220',label="SURF")
plt.plot(x,k5,color = '#6950a1',label="ORB")

plt.title('The Number of Feature Points Detected at Different Time')
plt.xticks(x)
plt.xlabel("Hour")
plt.ylabel("Number of Feature Points")
plt.legend(loc = "best")
plt.show()

