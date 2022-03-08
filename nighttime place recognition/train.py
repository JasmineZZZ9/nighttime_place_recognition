from ast import Num
import os
from statistics import median
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools as it

IMG_HEIGHT = 464
IMG_WIDTH = 640

folder_name = []

lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, 0.6) * 255.0, 0, 255)

# corner detect
def applyCorner_SIFT_BF(path1, path2):
    
    img_1 = cv2.imread(path1)
    img_2 = cv2.imread(path2)
    
    #img_1 = cv2.resize(img_1, (IMG_HEIGHT, IMG_WIDTH))
    #img_2 = cv2.resize(img_2, (IMG_HEIGHT, IMG_WIDTH))
    
    # Denoise
    img_1 = cv2.fastNlMeansDenoising(img_1)
    img_2 = cv2.fastNlMeansDenoising(img_2)
    # Gamma correction
    img_1 = cv2.LUT(img_1, lookUpTable)
    img_2 = cv2.LUT(img_2, lookUpTable)
            
    img_1_g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_g = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    
    # ptrGFTT
    ptrGFTT = cv2.GFTTDetector_create(qualityLevel=0.01, minDistance=3.0,
                                  blockSize=3, useHarrisDetector=True, k=0.04)
    kp1 = ptrGFTT.detect(img_1_g, None)
    kp2 = ptrGFTT.detect(img_2_g, None)

    sift = cv2.xfeatures2d.SIFT_create()

    des1 = sift.compute(img_1_g, kp1)
    des2 = sift.compute(img_2_g, kp2)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1[1],des2[1], k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 *n.distance:
            good.append([m])

    print('Corner_SIFT_BF:', len(good))
    num_matches = len(good)

    return num_matches

#def applyPtrGFTT_FLANN(path1, path2):
#    
#    img_1 = cv2.imread(path1)
#    img_2 = cv2.imread(path2)
#
#    img_1 = cv2.resize(img_1, (IMG_HEIGHT, IMG_WIDTH))
#    img_2 = cv2.resize(img_2, (IMG_HEIGHT, IMG_WIDTH))
#            
#    img_1_g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
#    img_2_g = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
#
#    ptrGFTT = cv2.GFTTDetector_create(qualityLevel=0.01, minDistance=3.0,
#                                  blockSize=3, useHarrisDetector=True, k=0.04)
#    kp1 = ptrGFTT.detect(img_1_g, None)
#    kp2 = ptrGFTT.detect(img_2_g, None)
#
#    sift = cv2.xfeatures2d.SIFT_create()
#
#    des1 = sift.compute(img_1_g, kp1)
#    des2 = sift.compute(img_2_g, kp2)
#
#    # FLANN
#    FLANN_INDEX_KDTREE = 0
#    index_params = dict(algorithm = FLANN_INDEX_KDTREE,
#                    trees = 5)
#    search_params = dict(check = 50)
#
#    flann = cv2.FlannBasedMatcher(index_params, search_params)
#    matches = flann.knnMatch(des1[1], des2[1], k=2)
#    
#    num_matches = len(matches)
#    print('PtrGFTT_FLANN:', len(matches))
#    return num_matches


# SIFT
#def applySIFT_BF(path1, path2):
#    
#    img_1 = cv2.imread(path1)
#    img_2 = cv2.imread(path2)
#
#    img_1 = cv2.resize(img_1, (IMG_HEIGHT, IMG_WIDTH))
#    img_2 = cv2.resize(img_2, (IMG_HEIGHT, IMG_WIDTH))
#            
#    img_1_g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
#    img_2_g = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
#
#    sift = cv2.xfeatures2d.SIFT_create()
#
#    kp1, des1 = sift.detectAndCompute(img_1_g, None)
#    kp2, des2 = sift.detectAndCompute(img_2_g, None)
#
#    bf = cv2.BFMatcher_create(cv2.NORM_L1)
#    matches = bf.match(des1, des2)
#    
#    num_matches = len(matches)
#    print('SIFT_BF:', len(matches))
#    return num_matches

#def applySIFT_FLANN(path1, path2):
#    
#    img_1 = cv2.imread(path1)
#    img_2 = cv2.imread(path2)
#
#    img_1 = cv2.resize(img_1, (IMG_HEIGHT, IMG_WIDTH))
#    img_2 = cv2.resize(img_2, (IMG_HEIGHT, IMG_WIDTH))
#
#    img_1_g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
#    img_2_g = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
#
#    sift = cv2.xfeatures2d.SIFT_create()
#
#    kp1, des1 = sift.detectAndCompute(img_1_g, None)
#    kp2, des2 = sift.detectAndCompute(img_2_g, None)
#
#    # FLANN
#    FLANN_INDEX_KDTREE = 0
#    index_params = dict(algorithm = FLANN_INDEX_KDTREE,
#                    trees = 5)
#    search_params = dict(check = 50)
#
#    flann = cv2.FlannBasedMatcher(index_params, search_params)
#    matches = flann.knnMatch(des1, des2, k=2)
#    good = []
#    for i, (m, n) in enumerate(matches):
#        if m.distance < 0.7 * n.distance:
#            good.append(m)
#
#    num_matches = len(good)
#    print('SIFT_FLANN:', len(good))
#    return num_matches

# SURF
#def applySURF_BF(path1, path2):
#    
#    img_1 = cv2.imread(path1)
#    img_2 = cv2.imread(path2)
#
#    img_1 = cv2.resize(img_1, (IMG_HEIGHT, IMG_WIDTH))
#    img_2 = cv2.resize(img_2, (IMG_HEIGHT, IMG_WIDTH))
#
#    img_1_g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
#    img_2_g = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
#
#    surf = cv2.xfeatures2d.SURF_create()
#    kp1,des1 = surf.detectAndCompute(img_1_g, None)
#    kp2,des2 = surf.detectAndCompute(img_2_g, None)
#
#    bf = cv2.BFMatcher_create(cv2.NORM_L1)
#    matches = bf.match(des1, des2)
#    
#    num_matches = len(matches)
#    print('SURF_BF:', len(matches))
#    return num_matches

# ORB
#def applyORB_BF(path1, path2):
#    
#
#    img_1 = cv2.imread(path1)
#    img_2 = cv2.imread(path2)
#
#    img_1 = cv2.resize(img_1, (IMG_HEIGHT, IMG_WIDTH))
#    img_2 = cv2.resize(img_2, (IMG_HEIGHT, IMG_WIDTH))
#
#    img_1_g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
#    img_2_g = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
#
#    orb = cv2.ORB_create()
#    kp1,des1 = orb.detectAndCompute(img_1_g, None)
#    kp2,des2 = orb.detectAndCompute(img_2_g, None)
#
#    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=False)
#    matches = bf.match(des1, des2)
#
#    num_match = len(matches)
#    print('ORB_BF:', len(matches))
#    return num_match

#def applyORB_KNN(path1, path2):
#    
#    img_1 = cv2.imread(path1)
#    img_2 = cv2.imread(path2)
#
#    img_1 = cv2.resize(img_1, (IMG_HEIGHT, IMG_WIDTH))
#    img_2 = cv2.resize(img_2, (IMG_HEIGHT, IMG_WIDTH))
#
#    img_1_g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
#    img_2_g = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
#
#    orb = cv2.ORB_create()
#    kp1,des1 = orb.detectAndCompute(img_1_g, None)
#    kp2,des2 = orb.detectAndCompute(img_2_g, None)
#
#    bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
#
#    # knnMatch
#    matches = bf.knnMatch(des1,des2,k=1)
#    print('ORB_KNN:', len(matches))
#    num_match = len(matches)
#    
#    return num_match

def read_path(file_pathname):
    path = []
    for filename in os.listdir(file_pathname):
        if '.jpg' in filename:
            #print(filename)
            path.append(file_pathname+'/'+filename)
        else:
            continue
    return path
        
def read_path_all(path):
    for filename in os.listdir(path):
        if '.' not in filename:
            file_path = os.path.join(path, filename)
            folder_name.append(file_path)
        else:
            continue

read_path_all('/Users/zengmingjie/Documents/Assignment/digital image processing/midterm/nighttime_place_recognition/nighttime place recognition dataset/train')
#print(folder_name)
print('There are', len(folder_name), 'place folders.')

path1 = read_path(folder_name[5])
path2 = read_path(folder_name[8])
#print(path1)
print('There are', len(path1), 'files in path1.')
print('There are', len(path2), 'files in path2.')

#path1 = ['midterm/nighttime_place_recognition/nighttime place recognition dataset/train/00005381/20151102_101043.jpg','midterm/nighttime_place_recognition/nighttime place recognition dataset/train/00005381/20151102_211038.jpg']
combinations = it.product(path1, path2)
#print('combinations:',combinations)

sum = len(list(combinations))
print('There are', sum, "pairs of images.")
num_match_same = np.zeros((1,sum))
flag = 0

#num_match_same = np.zeros((1,4))
for i in it.product(path1, path2):
    #if flag < 4:
    #    print('pair:', i)
    #    num_match_same[0][flag] = applyCorner_SIFT_BF(i[0], i[1])
    #else:
    #    break
    print('pair:', i)
    num_match_same[0][flag] = applyCorner_SIFT_BF(i[0], i[1])
    #num_match_same[1][flag] = applyPtrGFTT_FLANN(i[0], i[1])
    #num_match_same[2][flag] = applyORB_BF(i[0], i[1])
    #num_match_same[3][flag] = applyORB_KNN(i[0], i[1])
    #num_match_same[4][flag] = applySIFT_BF(i[0], i[1])
    #num_match_same[5][flag] = applySIFT_FLANN(i[0], i[1])
    #num_match_same[6][flag] = applySURF_BF(i[0], i[1])
    flag += 1
#print(num_match_same)
x = range(sum)
#x = range(4)
k1 = num_match_same[0][:]
#k2 = num_match_same[1][:]
#k3 = num_match_same[2][:]
#k4 = num_match_same[3][:]
#k5 = num_match_same[4][:]
#k6 = num_match_same[5][:]
#k7 = num_match_same[6][:]

# mean median count min max
mean = np.mean(num_match_same)
media = np.median(num_match_same)
percent = np.percentile(num_match_same, 85)
min = np.min(num_match_same)
max = np.max(num_match_same)

print('mean:', mean)
print('media:', media)
print('percent:', percent)
print('min:', min)
print('max:', max)

plt.scatter(x,k1,s=5., color = '#00ae9d',label="Corner_SIFT_BF", alpha=0.7)
#plt.plot(x,k2,color = '#f7acbc',marker='+',label="PtrGFTT_FLANN")
#plt.plot(x,k3,color = '#7f7522',marker='*',label="ORB_BF")
#plt.plot(x,k4,color = '#f58220',marker='1',label="ORB_KNN")
#plt.plot(x,k5,color = '#6950a1',marker='x',label="SIFT_BF")
#plt.plot(x,k6,color = '#b22c46',marker='.',label="SIFT_FLANN")
#plt.plot(x,k7,color = '#ffd400',marker='s',label="SURF_BF")

plt.title('The Number of Matches Detected at Different Pair of Same Place')
plt.xlabel("#Pair")
plt.ylabel("Number of Matches")
plt.legend(loc = "best")
plt.show()

