from ast import Num
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
from prettytable import PrettyTable

IMG_HEIGHT = 464
IMG_WIDTH = 640

folder_name = []

place_1 = '00021510'
place_2 = '00023966'

# metric
TP = 0
FP = 0
TN = 0
FN = 0

lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, 0.6) * 255.0, 0, 255)

# corner detect
def testCorner_SIFT_BF(path1, path2):
    
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
    threshold = 5

    if len(good) > threshold:
        predict = 1
    else:
        predict = 0
    
    print('Corner_SIFT_BF:', len(good))
    #img_3 = cv2.drawMatchesKnn(img_1,kp1,img_2,kp2, good, None)
    return predict

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

read_path_all('/Users/zengmingjie/Documents/Assignment/digital image processing/midterm/nighttime_place_recognition/nighttime place recognition dataset/test')
#print(folder_name)
print('There are', len(folder_name), 'place folders.')

path1 = read_path(folder_name[0])
path2 = read_path(folder_name[1])
#print(path1)
print('There are', len(path1), 'files in path1.')
print('There are', len(path2), 'files in path2.')
path = path1 + path2
path = path
#path1 = ['midterm/nighttime_place_recognition/nighttime place recognition dataset/train/00010888/20151102_015550.jpg','midterm/nighttime_place_recognition/nighttime place recognition dataset/train/00010888/20151102_105546.jpg']
combinations = it.product(path, path)
#print('combinations:',combinations)

sum = len(list(combinations))
#num_match_same = np.zeros((1,sum))
##flag = 0

for i in it.product(path, path):
    predict = testCorner_SIFT_BF(i[0], i[1])
    if predict == 1:
        if place_1 in i[0] and place_1 in i[1]:
            TP += 1
        elif place_2 in i[0] and place_2 in i[1]:
            TP += 1
        else:
            FP += 1
    elif predict == 0:
        if place_1 in i[0] and place_2 in i[1]:
            TN += 1
        elif place_2 in i[0] and place_1 in i[1]:
            TN += 1
        else:
            FN += 1

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
print('TP:', TP)
print('TF:', TN)
print('FP:', FP)
print('FN:', FN)

precision = TP / (TP +FP)
recall = TP / (TP + FN)
accuracy = (TP + TN) / (TP + TN + FP + FN)
f1 = 2 * recall * precision / (recall + precision)

table1 = PrettyTable(['', 'Matching Negative', 'Matching Positive'])
table1.add_row(['Actrual Negtive', TN, FP])
table1.add_row(['Actual Positive', FN, TP])
print(table1)

table2 = PrettyTable(['Precision','Recall','Accuracy','F1'])
table2.add_row([precision, recall, accuracy, f1])
print(table2)

cv2.waitKey(0)
cv2.destroyAllWindows()



