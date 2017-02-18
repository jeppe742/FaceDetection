# coding=utf-8
import cv2
import numpy as np
#import pyximport; pyximport.install()
#import c_hog
from skimage import color
from skimage import transform
from skimage.feature import hog
import os
import matplotlib.pyplot as plt
from sklearn import svm
#from skil
#import Image
#import hog

def slide_window(img, step_size, window_size):
    for y in range(0, img.shape[0]-window_size[0], step_size):
        for x in range(0, img.shape[1]-window_size[1], step_size):
            yield (x, y, img[y:y+window_size[0], x:x+window_size[1]])

data=[]
result=[]

for file in os.listdir("./images/Train_face"):
    img = np.asarray(cv2.imread(os.path.join("./images/Train_face", file), 0), dtype=int)
    if img is not None:
        print("Arbejder på face : ", file)
        hist = hog(img,orientations=4, pixels_per_cell=(4,4),cells_per_block=(1,1))
        data.append(hist)
        result.append(1)

for file in os.listdir("./images/Train_NoFace"):
    img = np.asarray(cv2.imread(os.path.join("./images/Train_NoFace", file),1))
    img = color.rgb2gray(img)
    if img is not None:
        print("Arbejder on noface: ", file, " size= ", img.shape)
        for pyramid in transform.pyramid_gaussian(img,downscale=2):
            i=0
            for (x,y,box) in slide_window(pyramid,10,(36,36)):

                # if i==5:
                #     i=0
                #     img_tmp=pyramid.copy()
                #     cv2.rectangle(img_tmp,(x,y),(x+36, y+ 36),(0,255,255),2)
                #     cv2.imshow("window",img_tmp)
                #     cv2.waitKey(1)
                # i+=1
                
                hist = hog(box,orientations=4, pixels_per_cell=(4,4),cells_per_block=(1,1))
                data.append(hist)
                result.append(0)

data = np.asarray(data)
result = np.asarray(result)
clf = svm.SVC(verbose=True)
clf.fit(data,result)
clf.score(data,result)

print("done")
# #def main():
# cap = cv2.VideoCapture(0)
# #for i in range(10):
# while True:
#     _,img = cap.read()
#     img = color.rgb2gray(img)
#     img = transform.downscale_local_mean(img,(8,6))
#     window=(36,36)
#     for (x,y,box) in slide_window(img,18,window):

#         cv2.rectangle(img,(x,y),(x+window[0], y+ window[1]),(0,255,0),2)
#         cv2.imshow("window",img)
#         cv2.waitKey(1)


# cap.release()