import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def is_o(n):
    return n != 0

def get_pig_weight(pig_weight_list):
    pig_weight_list = list(filter(is_o, pig_weight_list))
    return pig_weight_list

def area_ratio_transform(x):
    if x == 'small':
        return 0
    elif x == 'normal':
        return 1
    else:
        return 2

def color_transform(x):
    if x == 'white':
        return 0
    elif x == 'light_color':
        return 1
    elif x == 'black':
        return 2
    else:  # dirty
        return 3

def get_csv_info_():
    # fig = plt.figure()
    # ax = Axes3D(fig)
    csv_path = './pig_statistics_new.csv'
    with open(csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)  # 读取csv文件，返回的是迭代类型
        # print(reader2)
        pig_length_list = []
        pig_weight_list = []
        photo_distance_list = []
        pig_color_list = []
        pig_angle_list=[]
        pig_ID_list=[]
        for i, csv_list in enumerate(reader):
            if i:
                pig_ID=csv_list[1]
                pig_ID_list.append(pig_ID)

                pig_length = int(float(csv_list[5].split('cm')[0]))
                pig_length_list.append(pig_length)

                pig_weight = int(float(csv_list[6].split('kg')[0]))
                pig_weight_list.append(pig_weight)

                photo_distance = csv_list[9]
                photo_distance_list.append(photo_distance)

                pig_color = csv_list[7]
                pig_color_list.append(pig_color)

                pig_angle=csv_list[10]
                pig_angle_list.append(pig_angle)

        photo_distance_list = list(map(area_ratio_transform, photo_distance_list))
        pig_color_list = list(map(color_transform, pig_color_list))
        """长度直方图"""
        # plt.hist(pig_length_list, bins=max(pig_length_list), color='r')
        # plt.xlabel('length')
        # plt.ylabel('photo_count')
        # plt.show()
        # """夹角直方图"""
        # pig_angle_list=list(map(float,pig_angle_list))
        # plt.hist(pig_angle_list, bins=int(max(pig_angle_list)), color='b')
        # plt.xlabel('angle')
        # plt.ylabel('photo_count')
        # plt.show()
        # """猪的种类与夹角直方图"""
        # pig_ID_list=list(map(int,pig_ID_list))
        # r=plt.hist2d(pig_ID_list, pig_angle_list,bins=(30,30),cmap=plt.cm.jet)
        # plt.xlabel('pig_ID')
        # plt.ylabel('angle')
        # plt.colorbar(r[3])
        # plt.show()
        # """长度重量直方图"""
        # h=plt.hist2d(pig_length_list, pig_weight_list,bins=(25,25),cmap=plt.cm.jet)
        # plt.xlabel('length')
        # plt.ylabel('weight')
        # plt.colorbar(h[3])
        # plt.show()
        # """重量直方图"""
        pig_weight_list=get_pig_weight(pig_weight_list)
        plt.hist(pig_weight_list, bins=max(pig_weight_list)+100,color='g')
        plt.xlabel('weight')
        plt.ylabel('photo_count')
        plt.show()
        # """猪面积与照片面积比直方图"""
        # plt.hist(photo_distance_list,bins=3,color='b')
        # plt.xlabel('pig_photo_ratio')
        # plt.ylabel('photo_count')
        # plt.xticks((0.5, 1, 1.5), ('small', 'normal', 'big'))
        # plt.show()
        # """猪颜色直方图"""
        # plt.hist(pig_color_list,bins=4, color='y')
        # plt.xlabel('pig_color')
        # plt.ylabel('photo_count')
        # plt.xticks((0.5, 1,2,2.5), ('white', 'light_color','black','dirty'))
        # plt.show()
def pig_csv_test_csv_merge():
    pig_csv_path = './pig_statistics_new.csv'
    test_error_csv_path='./pig_dev_pro_data.csv'
    with open(pig_csv_path, "r") as csvfile:
        reader = csv.reader(csvfile)  # 读取csv文件，返回的是迭代类型
        for i, csv_list in enumerate(reader):
            if i:
                with open(test_error_csv_path, "r") as csvfile:
                    reader = csv.reader(csvfile)  # 读取csv文件，返回的是迭代类型
                    for i, csv_list in enumerate(reader):
                        if i:
                            pass

if __name__ == '__main__':
    # get_csv_info_()
    pig_csv_test_csv_merge()