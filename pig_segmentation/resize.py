#coding: utf-8
#resize image for training.

import os
import cv2
import glob

#folder
img_list  = os.listdir('./dataset/guiyang_data/testlabel')


for i in img_list:
    if i[-4:] != '.DS_Store':
        print(i)
        #img = cv2.imread('./dataset/guiyang_data/testlabel/' + i)
        #_,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        #img = cv2.resize(img,(768,768))
        
        os.rename('./dataset/guiyang_data/testlabel/'+i,'./dataset/guiyang_data/testlabel/'+i[:14] + '.jpg')
        #os.rename('./dataset/guiyang_data/testlabel/' + i, './dataset/guiyang_data/testlabel/' + i[:14] + '.jpg')
        #os.rename('./dataset/guiyang_data/testlabel/' + i, './dataset/guiyang_data/testlabel/' + i[:14] + '.jpg')
        #cv2.imwrite('./dataset/guiyang_data/testlabel/' + str(i), img)

        print('Ok')
