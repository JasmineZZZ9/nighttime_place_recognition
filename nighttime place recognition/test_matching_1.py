from ast import Num
import os
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
    

    img_1 = cv2.resize(img_1, (IMG_HEIGHT, IMG_WIDTH))
    img_2 = cv2.resize(img_2, (IMG_HEIGHT, IMG_WIDTH))
    
    # Denoise
    img_1 = cv2.fastNlMeansDenoising(img_1)
    img_2 = cv2.fastNlMeansDenoising(img_2)

    # Gamma correction
    img_1 = cv2.LUT(img_1, lookUpTable)
    img_2 = cv2.LUT(img_2, lookUpTable)
            
    img_1_g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_g = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    
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

    img_3 = cv2.drawMatchesKnn(img_1, kp1, img_2, kp2,
                             good, None)

    print('Corner_SIFT_BF:', len(good))
    #img_3 = cv2.drawMatchesKnn(img_1,kp1,img_2,kp2, good, None)

    cv2.imshow('Corner_SIFT_BF result', img_3)
    cv2.imwrite('Corner_SIFT_BF_3.jpg', img_3)
    

# check 1
def applyCorner_SIFT_FLANN(path1, path2):
    
    img_1 = cv2.imread(path1)
    img_2 = cv2.imread(path2)

    img_1 = cv2.resize(img_1, (IMG_HEIGHT, IMG_WIDTH))
    img_2 = cv2.resize(img_2, (IMG_HEIGHT, IMG_WIDTH))

    # Denoise
    img_1 = cv2.fastNlMeansDenoising(img_1)
    img_2 = cv2.fastNlMeansDenoising(img_2)

    # Gamma correction
    img_1 = cv2.LUT(img_1, lookUpTable)
    img_2 = cv2.LUT(img_2, lookUpTable)
            
    img_1_g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_g = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    ptrGFTT = cv2.GFTTDetector_create(qualityLevel=0.01, minDistance=3.0,
                                  blockSize=3, useHarrisDetector=True, k=0.04)
    kp1 = ptrGFTT.detect(img_1_g, None)
    kp2 = ptrGFTT.detect(img_2_g, None)

    sift = cv2.xfeatures2d.SIFT_create()

    des1 = sift.compute(img_1_g, kp1)
    des2 = sift.compute(img_2_g, kp2)

    # FLANN
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1[1], des2[1], k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            good.append([m])

    print('Corner_SIFT_FLANN:', len(good))
    img_3 = cv2.drawMatchesKnn(img_1,kp1,img_2,kp2, good, None)

    cv2.imshow('Corner_SIFT_FLANN result', img_3)
    cv2.imwrite('Corner_SIFT_FLANN_3.jpg', img_3)
    

# SIFT check 1
def applySIFT_BF(path1, path2):
    
    img_1 = cv2.imread(path1)
    img_2 = cv2.imread(path2)

    img_1 = cv2.resize(img_1, (IMG_HEIGHT, IMG_WIDTH))
    img_2 = cv2.resize(img_2, (IMG_HEIGHT, IMG_WIDTH))

    # Denoise
    img_1 = cv2.fastNlMeansDenoising(img_1)
    img_2 = cv2.fastNlMeansDenoising(img_2)

    # Gamma correction
    img_1 = cv2.LUT(img_1, lookUpTable)
    img_2 = cv2.LUT(img_2, lookUpTable)
            
    img_1_g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_g = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img_1_g, None)
    kp2, des2 = sift.detectAndCompute(img_2_g, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7 *n.distance:
            good.append([m])

    print('SIFT_BF:', len(good))
    img_3 = cv2.drawMatchesKnn(img_1, kp1, img_2, kp2, good, None)

    cv2.imshow('SIFT_BF result', img_3)
    cv2.imwrite('SIFT_BF_3.jpg', img_3)
    
    
# check 1
def applySIFT_FLANN(path1, path2):
    
    img_1 = cv2.imread(path1)
    img_2 = cv2.imread(path2)

    img_1 = cv2.resize(img_1, (IMG_HEIGHT, IMG_WIDTH))
    img_2 = cv2.resize(img_2, (IMG_HEIGHT, IMG_WIDTH))

    # Denoise
    img_1 = cv2.fastNlMeansDenoising(img_1)
    img_2 = cv2.fastNlMeansDenoising(img_2)

    # Gamma correction
    img_1 = cv2.LUT(img_1, lookUpTable)
    img_2 = cv2.LUT(img_2, lookUpTable)

    img_1_g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_g = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img_1_g, None)
    kp2, des2 = sift.detectAndCompute(img_2_g, None)

    # FLANN
    index_params = dict(algorithm = 0,
                    trees = 5)
    search_params = dict(check = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            good.append(m)
    img_3 = cv2.drawMatchesKnn(img_1, kp1, img_2, kp2, [good], None)
    print('SIFT_FLANN:', len(good))

    cv2.imshow('SIFT_FLANN result', img_3)
    cv2.imwrite('SIFT_FLANN_3.jpg', img_3)
    

# SURF # check 1
def applySURF_BF(path1, path2):
    
    img_1 = cv2.imread(path1)
    img_2 = cv2.imread(path2)

    img_1 = cv2.resize(img_1, (IMG_HEIGHT, IMG_WIDTH))
    img_2 = cv2.resize(img_2, (IMG_HEIGHT, IMG_WIDTH))

    # Denoise
    img_1 = cv2.fastNlMeansDenoising(img_1)
    img_2 = cv2.fastNlMeansDenoising(img_2)

    # Gamma correction
    img_1 = cv2.LUT(img_1, lookUpTable)
    img_2 = cv2.LUT(img_2, lookUpTable)

    img_1_g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_g = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    surf = cv2.xfeatures2d.SURF_create(float(4000))
    kp1,des1 = surf.detectAndCompute(img_1_g, None)
    kp2,des2 = surf.detectAndCompute(img_2_g, None)

    bf = cv2.BFMatcher(normType=cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    print('SURF_BF:', len(matches))
    img_3 = cv2.drawMatches(img_1, kp1, img_2, kp2, matches, None)

    cv2.imshow('SURF_BF result', img_3)
    cv2.imwrite('SURF_BF_3.jpg', img_3)
    
    
# ORB # check 1
def applyORB_BF(path1, path2):
    

    img_1 = cv2.imread(path1)
    img_2 = cv2.imread(path2)

    img_1 = cv2.resize(img_1, (IMG_HEIGHT, IMG_WIDTH))
    img_2 = cv2.resize(img_2, (IMG_HEIGHT, IMG_WIDTH))

    # Denoise
    img_1 = cv2.fastNlMeansDenoising(img_1)
    img_2 = cv2.fastNlMeansDenoising(img_2)

    # Gamma correction
    img_1 = cv2.LUT(img_1, lookUpTable)
    img_2 = cv2.LUT(img_2, lookUpTable)

    img_1_g = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_g = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1,des1 = orb.detectAndCompute(img_1_g, None)
    kp2,des2 = orb.detectAndCompute(img_2_g, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    print('ORB_BF:', len(matches))
    img_3 = cv2.drawMatches(img_1, kp1, img_2, kp2, matches, None)

    cv2.imshow('ORB_BF result', img_3)
    cv2.imwrite('ORB_BF_3.jpg', img_3)
    
    
# check 1
#def applyORB_KNN(path1, path2):
#    
#    img_1 = cv2.imread(path1)
#    img_2 = cv2.imread(path2)
#
#    img_1 = cv2.resize(img_1, (IMG_HEIGHT, IMG_WIDTH))
#    img_2 = cv2.resize(img_2, (IMG_HEIGHT, IMG_WIDTH))
#
#    # Denoise
#    img_1 = cv2.fastNlMeansDenoising(img_1)
#    img_2 = cv2.fastNlMeansDenoising(img_2)
#
#    # Gamma correction
#    img_1 = cv2.LUT(img_1, lookUpTable)
#    img_2 = cv2.LUT(img_2, lookUpTable)
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
#    img3 = cv2.drawMatchesKnn(img_1,kp1,img_2,kp2, matches, None)
#
#    cv2.imshow('ORB_KNN result', img3)
    
    

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

#path1 = read_path(folder_name[0])
#path2 = folder_name[2]
#print(path1)
#print('There are', len(path1), 'files.')

path1 = ['midterm/nighttime_place_recognition/nighttime place recognition dataset/train/00010888/20151102_015550.jpg','midterm/nighttime_place_recognition/nighttime place recognition dataset/train/00010888/20151102_105546.jpg']
#combinations = it.product(path1, path1)
#print('combinations:',combinations)

#sum = len(list(combinations))
#num_match_same = np.zeros((7,sum))
#flag = 0

#for i in it.product(path1, path1):
    #print('combination:', i)
    #num_match_same[0][flag] = applyPtrGFTT_BF(i[0], i[1])
    #num_match_same[1][flag] = applyPtrGFTT_FLANN(i[0], i[1])
    #num_match_same[2][flag] = applyORB_BF(i[0], i[1])
    #num_match_same[3][flag] = applyORB_FLANN(i[0], i[1])
    #num_match_same[4][flag] = applySIFT_BF(i[0], i[1])
    #num_match_same[5][flag] = applySIFT_FLANN(i[0], i[1])
    #num_match_same[6][flag] = applySURF_BF(i[0], i[1])
    #applyPtrGFTT_BF(i[0], i[1])
#
applyCorner_SIFT_BF(path1[0], path1[1])
applyCorner_SIFT_FLANN(path1[0], path1[1])
#
applyORB_BF(path1[0], path1[1])
#
#applyORB_KNN(path1[0], path1[1])
#
applySIFT_BF(path1[0], path1[1])
#
applySIFT_FLANN(path1[0], path1[1])
#
applySURF_BF(path1[0], path1[1])


cv2.waitKey(0)
cv2.destroyAllWindows()



