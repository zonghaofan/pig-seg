#coding:utf-8
import os
import cv2
from math import *
import numpy as np
import shutil
import imutils

def train_test_image():
    out_put='./back'
    if not os.path.exists(out_put):
        os.mkdir(out_put)
    input='./back_label'
    for i in os.listdir(input):
        if i in os.listdir('./train'):
            img=cv2.imread('./train'+'/'+i)
            cv2.imwrite(out_put+'/'+i,img)
        elif i in os.listdir('./test'):
            img = cv2.imread('./test' + '/' + i)
            cv2.imwrite(out_put + '/' + i, img)
def rotate_image(img, height, width, degree=90):
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    img_rotated = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return img_rotated
#添加背景为负样本
def procss_back_resize():
    path = './background'
    out_path = './background_out'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    images_list_pig = [os.path.join(path, dir) for dir in os.listdir(path)]
    # print(images_list_pig)
    for num, image_list_pig in enumerate(images_list_pig):
        print(image_list_pig)
        if '.jpg' in image_list_pig:
            image = cv2.imread(image_list_pig)
            h, w = image.shape[:2]
            if h > w:
                image = rotate_image(image, h, w)
            image = cv2.resize(image, (1600, 1200))
            cv2.imwrite(out_path + '/'+ image_list_pig.split('/')[-1], image)
            print('{}image'.format(num))
#输出纯黑图片,制作负样本
def black_image_out():
    image_path='./back'

    output_path='./backlabel'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    images_path_list=[os.path.join(image_path,i) for i in os.listdir(image_path)]
    for i in images_path_list:
        image=cv2.imread(i)
        height,width=image.shape[:2]
        black_mat=np.zeros(shape=[height,width,3],dtype=np.float32)
        cv2.imwrite(output_path+'/'+i.split('/')[-1],black_mat)
#根据10个文件夹的名字找到对应的label写入10个label文件夹
def image_match_label():
    # 生成10个label文件夹
    for i in range(1, 11):
        out_put_dir = './data/train' + str(i)+'label'
        if not os.path.exists(out_put_dir):
            os.mkdir(out_put_dir)
    for i in range(1, 11):
        pig_raw_path='./data/train'+str(i)
        pig_mask_path='./data/pig_mask'
        pig_label_path='./data/train' + str(i)+'label'
        for count,j in enumerate(os.listdir(pig_raw_path)):
            if j in os.listdir(pig_mask_path):
                pig_mask=cv2.imread(pig_mask_path+'/'+j)
                cv2.imwrite(pig_label_path+'/'+j,pig_mask)
                print('{}image'.format(i*count))
#将背景和背景的label分别加入到10个文件夹
def back_split_dirs():
    input_path='./waibao_image'
    back_images_list=[os.path.join(input_path,i) for i in os.listdir(input_path)]
    for count,i in enumerate(back_images_list):
        img = cv2.imread(i)
        img_label=cv2.imread(i.replace('waibao_image','waibao_label'))
        index=np.random.randint(1, 11, 1)
        cv2.imwrite('./train' + str(index[0])+'/'+i.split('/')[-1], img)
        cv2.imwrite('./train' + str(index[0]) + 'label/' + i.split('/')[-1], img_label)
        print('{}image'.format(count))
#删除10个训练文件夹和label不需要的文件
def del_no_need_train10():
    for i in range(1, 11):
        need_path='./train'+str(i)

        del_path='./train'+str(i)+'label'

        need_names=os.listdir(need_path)
        del_names=os.listdir(del_path)

        for i in del_names:
            if i in need_names:
                pass
            else:
                del_path_name=os.path.join(del_path, i)
                os.remove(del_path_name)
#将多个文件夹图片合成一个
def merge_dirs_photo():
    output_raw_path='./raw'
    output_pig_mask_path = './pig_mask'
    if not (os.path.exists(output_raw_path) or os.path.exists(output_pig_mask_path)):
        os.mkdir(output_pig_mask_path)
        os.mkdir(output_raw_path)
    for i in range(1,11):
        input_path='./train'+str(i)
        images_path_list=[os.path.join(input_path,i) for i in os.listdir(input_path)]
        for image_path_list in images_path_list:
            # shutil.copy(image_path_list,output_raw_path)
            shutil.copy(image_path_list.replace('train'+str(i),'train'+str(i)+'label'), output_pig_mask_path)
#由mask的猪裁解原图
def mask_pig_resize():
    pig_path='./raw'
    pig_mask_path='./pig_mask'
    out_path='./pig_mask_resize'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    images_path_list=[os.path.join(pig_path,i) for i in os.listdir(pig_path)]
    for count,image_path_list in enumerate(images_path_list):
        if 'length' in image_path_list:
            pig=cv2.imread(image_path_list)
            # print(pig.shape)
            pig_mask=cv2.imread(image_path_list.replace('raw','pig_mask'))
            pig_mask_gray=cv2.cvtColor(pig_mask,cv2.COLOR_BGR2GRAY)
            pig_mask_th=cv2.threshold(pig_mask_gray,127,255,cv2.THRESH_BINARY)[1]
            x,y,w,h=cv2.boundingRect(pig_mask_th)

            # cv2.rectangle(pig_mask, (x, y), (x + w, y + h), (255, 255, 255),5)
            # cv2.imshow('11',pig_mask)
            # cv2.waitKey(0)

            pig_leave=(pig_mask/255)*pig
            pig_leave = pig_leave[y:y + h, x:x + w]
            cv2.imwrite(out_path+'/'+image_path_list.split('/')[-1], pig_leave)
            print('{}image'.format(count))
#根据分割的结果制作label
def accord_seg_label():
    img_path = './waibao_label'
    out_path = './waibao_label_out'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    images_list_path = [os.path.join(img_path, i) for i in os.listdir(img_path)]

    for num, image_list_path in enumerate(images_list_path):
        img = cv2.imread(image_list_path, cv2.IMREAD_GRAYSCALE)
        print(image_list_path)
        # 二值化找轮廓
        image_thre = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(image_thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if imutils.is_cv2() else cnts[1]
        c_ = sorted(contours, key=cv2.contourArea, reverse=True)
        pig_cnt = np.squeeze(c_[0])

        image_h = img.shape[0]
        image_w = img.shape[1]
        mask = np.zeros((image_h, image_w, 3))
        # plt.plot(pig_cnt[:, 0], pig_cnt[:, 1])
        # plt.show()
        dummy_mask = cv2.drawContours(mask, [pig_cnt], 0, (255, 255, 255), thickness=cv2.FILLED)
        cv2.imwrite(out_path + '/' + image_list_path.split('/')[-1], dummy_mask)
        print('{}image'.format(num))
#跟据图片找出对应的label
def find_pig_label():
    img_path='./bad_pig_mask'

    label_path='./pig_20181221_jiao_output'

    out_path='./bad_pig_mask_output'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    images_list_path=[os.path.join(img_path,i) for i in os.listdir(img_path)]
    print(images_list_path)
    for num,image_list_path in enumerate(images_list_path):
        if image_list_path.split('/')[-1] in os.listdir(label_path):

            img=cv2.imread(image_list_path.replace('bad_pig_mask','pig_20181221_jiao_output'))

            cv2.imwrite(out_path +'/'+image_list_path.split('/')[-1], img)
            print('{}image'.format(num))

def del_no_need():
    need_path='./waibao_image'

    del_path='./waibao_label'

    need_names=os.listdir(need_path)
    del_names=os.listdir(del_path)

    for i in del_names:
        if i in need_names:
            pass
        else:
            del_path_name=os.path.join(del_path, i)
            os.remove(del_path_name)
#给拍的照片添加日期
def add_date():
    path='./pig_20181221_jiao_out_label'
    # path='./pig_20181221_jiao_out_label'
    images_list_path=[os.path.join(path,i) for i in os.listdir(path)]
    for count,image_list_path in enumerate(images_list_path):
        new_name=image_list_path.split('/')[-1].replace('kg','kg_20181221')
        # print(new_name)
        img_path=image_list_path.replace(image_list_path.split('/')[-1],new_name)
        # print(img_path)
        # print(image_list_path)
        os.rename(image_list_path,img_path)

if __name__ == '__main__':
    # train_test_image()
    # procss_back_resize()
    # black_image_out()
    # image_match_label()
    back_split_dirs()
    # del_no_need_train10()
    # merge_dirs_photo()
    # mask_pig_resize()
    # accord_seg_label()
    # find_pig_label()
    # del_no_need()
    # add_date()