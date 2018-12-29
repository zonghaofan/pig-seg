import os
import csv
import numpy as np
import cv2
import imutils
import time
import pandas as pd
import matplotlib.pyplot as plt
import math
CSV_C0_NAME = 'file_name'
CSV_C1_NAME = 'pig_ID'
CSV_C2_NAME = 'photo_date'
CSV_C3_NAME = 'pig_hometown'
CSV_C4_NAME = 'pig_photo_number'
CSV_C5_NAME = 'pig_length'
CSV_C6_NAME = 'pig_weight'
CSV_C7_NAME = 'pig_color'
CSV_C8_NAME = 'pig_area'
CSV_C9_NAME = 'pig_photo_area_ratio'
CSV_C10_NAME = 'pig_photo_angle'
image_h=1200
image_w=1600
def location(month,day):
    if month == '9':
        photo_location = 'sichuan_leshan'
    elif month == '11' and (day in list(map(str, np.arange(21, 27)))):
        photo_location = 'shandong_sishui'
    elif month == '11' and (day in list(map(str, np.arange(1, 21))) or list(map(str, np.arange(27, 30)))):
        photo_location = 'shandong_junan'
    else:
        photo_location = 'other'
    return photo_location

def is_o(n):
    return n != 0
def get_pig_color(pig_mask_leave):
    pig_mask_leave=np.sum(pig_mask_leave,axis=-1)
    pig_mask_leave=pig_mask_leave.flatten()
    pig_mask_leave = list(filter(is_o, pig_mask_leave))
    color_aver=np.mean(pig_mask_leave)
    return color_aver
def color_class(pig_hsv,pig_mask):
    """white_pig>100
    light_color_pig=90~100
    dirty_pig = 70~90
    black_pig=<70"""
    pig_mask_leave = (pig_mask / 255) * pig_hsv
    pig_mask_leave_color = np.sum(pig_mask_leave) / np.sum((pig_mask / 255))
    if pig_mask_leave_color>=100:
        pig_color='white'
    # elif 100<pig_mask_leave_color<=100:
    #     pig_color = 'yellow'
    elif 90<pig_mask_leave_color<=100:
        pig_color='light_color'
    elif 70<pig_mask_leave_color<=90:
        pig_color='dirty'
    else:
        pig_color = 'black'
    return pig_color
def contour_area(pig_mask):
    pig_mask = cv2.cvtColor(pig_mask, cv2.COLOR_BGR2GRAY)
    pig_im = cv2.threshold(pig_mask, 127, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(pig_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    c = sorted(cnts, key=cv2.contourArea, reverse=True)
    c = np.squeeze(c[0])
    pig_area=cv2.contourArea(c)
    return pig_area
def get_pig_photo_ratio(pig_area):
    pig_area_ratio=pig_area/(image_h*image_w)
    # print(disk_area_ratio)
    if pig_area_ratio>0.25:
        pig_photo_ratio='big'
    elif 0.11<=pig_area_ratio<=0.25:
        pig_photo_ratio='normal'
    else:
        pig_photo_ratio='small'
    return pig_photo_ratio
def del_no_length(x):
    if 'cm' not in x:
        return None
    else:
        return x
def get_pig_angle(pig_mask):
    img = cv2.cvtColor(pig_mask,cv2.COLOR_BGR2GRAY)
    # 二值化找轮廓
    image_thre = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    height, width = image_thre.shape[:2]
    a = np.array([255])
    y, x = np.where(image_thre == a)
    index_list = np.stack((x, y), axis=-1)
    # plt.plot(index_list[:, 0], index_list[:, 1], 'o')
    # plt.show()
    [vx, vy, x, y] = cv2.fitLine(index_list, cv2.DIST_L2, 0, 0.01, 0.01)
    # b = int((-x * vy / vx) + y)
    # y=k*(cols-1)+b=(cols-1-x1)*k+y1
    # righty = int(((width - 1 - x) * vy / vx) + y)
    # plt.plot(index_list[:, 0], index_list[:, 1], 'o')
    # plt.plot((0, width - 1), (b, righty))
    # plt.show()
    # 求旋转角
    theta = (np.arctan(vy/vx) * 180 / math.pi)[0]
    if theta < 0 :
        theta = abs(theta)
    else:
        theta=180-theta
    return theta
if __name__=='__main__':
    # input_dir_name='./train11'
    raw_path = './raw'
    raw_list_names = [os.path.join(raw_path, raw_name) for raw_name in os.listdir(raw_path)]

    raw_list_names = sorted(raw_list_names)
    raw_list_names=list(filter(del_no_length,raw_list_names))
    raw_list_names=sorted(raw_list_names,key=lambda x:
    float(x.split('/')[-1].split('length_')[-1].split('_weight_')[0].split('cm')[0]))
    print(raw_list_names)
    fieldnames = [CSV_C0_NAME, CSV_C1_NAME, CSV_C2_NAME, CSV_C3_NAME, CSV_C4_NAME, CSV_C5_NAME, CSV_C6_NAME,CSV_C7_NAME,CSV_C8_NAME,CSV_C9_NAME,CSV_C10_NAME]
    pig_photo_num=[]
    with open('pig_statistics.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        pig_num = 0
        pig_ID = 0
        pig_last_length = '19cm'
        pig_last_weight = '0kg'
        for i,raw_list_name in enumerate(raw_list_names):
            start_time = time.time()
            photo_name = raw_list_name.split('/')[-1]
            month_day = photo_name.split('_2018')[-1].replace('_', '').replace('-', '')[:4].replace('I', '')
            pig_date = '2018' + month_day
            month = str(int(month_day[:-2]))
            day = str(int(month_day[-2:]))
            """get location"""
            photo_location = location(month, day)
            """get length and weight"""
            pig_length = photo_name.split('length_')[-1].split('_weight_')[0]
            pig_weight = photo_name.split('_weight_')[-1].split('_')[0]
            """get color"""
            pig = cv2.imread(raw_list_name)
            pig_hsv = cv2.cvtColor(pig, cv2.COLOR_BGR2HSV)
            pig_mask = cv2.imread(raw_list_name.replace(raw_path, 'pig_mask'))
            pig_color = color_class(pig_hsv, pig_mask)
            """get pig angle"""
            theta=get_pig_angle(pig_mask)
            print('{}image,image={}'.format(i,raw_list_name.split('/')[-1]))
            # print('theta=',theta)
            """get pig area"""
            pig_area = contour_area(pig_mask)
            """get pig photo ratio"""
            pig_photo_ratio = get_pig_photo_ratio(pig_area)
            # cv2.imwrite('./data/pig_mask_leave/'+raw_list_name.split('/')[-1],pig_mask_leave)
            """get pig id and every pig photo numbers"""
            if pig_last_length == pig_length and pig_last_weight == pig_weight:
                pig_num += 1
            else:
                pig_photo_num.append(pig_num)
                pig_num = 1
                pig_ID += 1
                pig_last_length = pig_length
                pig_last_weight = pig_weight
            csv_row = {CSV_C0_NAME: photo_name,
                       CSV_C1_NAME: pig_ID,
                       CSV_C2_NAME: pig_date,
                       CSV_C3_NAME: photo_location,
                       CSV_C4_NAME: pig_num,
                       CSV_C5_NAME: pig_length,
                       CSV_C6_NAME: pig_weight,
                       CSV_C7_NAME: pig_color,
                       CSV_C8_NAME: pig_area,
                       CSV_C9_NAME: pig_photo_ratio,
                       CSV_C10_NAME: theta
                       }

            writer.writerow(csv_row)
            print('every photo time={}'.format(time.time()-start_time))
        pig_photo_num.append(pig_num)
    #写入每只猪拍的照片
    pig_photo_num_new=[0]
    for i in pig_photo_num:
        for j in range(i):
            pig_photo_num_new.append(i)
    print(pig_photo_num_new)
    with open('pig_statistics.csv', 'r') as csvfile,open('pig_statistics_new.csv','w') as csvfile_out :
        reader=csv.reader(csvfile)
        writer = csv.DictWriter(csvfile_out, fieldnames=fieldnames)
        writer.writeheader()
        writer_=csv.writer(csvfile_out)
        for i, csv_list in enumerate(reader):
            if i:
                print(len(csv_list))
                csv_list[4]=str(pig_photo_num_new[i])
                writer_.writerow(csv_list)


