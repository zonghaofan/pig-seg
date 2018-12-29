#coding:utf-8

from __future__ import print_function
import os
import os.path as osp
import numpy as np
import cv2
import csv
import time
import imutils
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from multiprocessing.dummy import Pool as ThreadPool
import itertools

plot_flag = False
write_flag = True


def get_contour(im):
    cnts = cv2.findContours(im.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnt = np.squeeze(cnt[0])
    return cnt


def get_contour_centroid(cnt):
    m = cv2.moments(cnt)
    c_x = int(m['m10'] / m['m00'])
    c_y = int(m['m01'] / m['m00'])
    return c_x, c_y


def get_convex_hull_points(cnt):
    hull = ConvexHull(cnt)
    hull = hull.vertices.tolist()
    hull.append(hull[0])
    points = cnt[hull]
    return points


def get_points_inside_contour(cnt, image_h, image_w):
    mask = np.zeros((image_h, image_w, 3))
    mask = cv2.drawContours(mask, [cnt], 0, (1, 0, 0), thickness=cv2.FILLED)
    y, x = np.where(mask[:, :, 0] == 1)
    inside_points = np.stack((x, y), axis=-1)
    return mask, inside_points


def rotate_image(img, height, width, degree=90):
    heightNew = int(width * math.fabs(math.sin(math.radians(degree))) + height * math.fabs(math.cos(math.radians(degree))))
    widthNew = int(height * math.fabs(math.sin(math.radians(degree))) + width * math.fabs(math.cos(math.radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    img_rotated = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
    return img_rotated


def image_preprocessing(pig_mask, disk_mask, raw_img_bgr):
    """求猪、圆盘的轮廓、重心，如果猪脚向上，翻转轮廓

    :param pig_mask:
    :param disk_mask:
    :param raw_img_bgr:
    :return:
    """
    disk_mask = cv2.morphologyEx(disk_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    disk_mask = cv2.morphologyEx(disk_mask, cv2.MORPH_OPEN, kernel)
    disk_mask = cv2.morphologyEx(disk_mask, cv2.MORPH_CLOSE, kernel)
    pig_mask = cv2.morphologyEx(pig_mask, cv2.MORPH_OPEN, kernel)
    pig_mask = cv2.morphologyEx(pig_mask, cv2.MORPH_CLOSE, kernel)

    mask = (pig_mask - disk_mask) > 0
    raw_img_gray = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2GRAY)
    # raw_img_gray = cv2.equalizeHist(raw_img_gray)
    pig_color_graylevel = np.sum(raw_img_gray * mask)/np.sum(mask)
    raw_img_brightness = cv2.cvtColor(raw_img_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    pig_color_brightness = np.sum(raw_img_brightness * mask) / np.sum(mask)

    pig_cnt0 = get_contour(pig_mask)

    # ##### 判断猪的方向是否垂直于图片 是则旋转图片90度 #####
    # [vx,vy,x,y] = cv2.fitLine(pig_cnt, cv2.DIST_L2,0,0.01,0.01)
    # grad = abs(vy/vx)
    # if grad>1:
    #     pig_mask = np.rot90(pig_mask)
    #     disk_mask = np.rot90(disk_mask)
    #     pig_mask = cv2.resize(pig_mask, (1600, int(1600 * 4 / 3)))
    #     disk_mask = cv2.resize(disk_mask, (1600, int(1600 * 4 / 3)))
    #     pig_cnt = get_contour(pig_mask)
    # disk_cnt = get_contour(disk_mask)
    # image_h, image_w = pig_mask.shape
    # ##########

    ##### 判断猪的"主轴"方向 旋转图片使主轴处于水平方向 #####
    image_h, image_w = pig_mask.shape

    _, inside_points = get_points_inside_contour(pig_cnt0, image_h, image_w)
    [vx, vy, x, y] = cv2.fitLine(inside_points, cv2.DIST_L2, 0, 0.01, 0.01)
    degree = np.arctan2(vy, vx) * 180 / math.pi
    degree = degree[0]
    # print(degree)
    pig_mask = rotate_image(pig_mask, image_h, image_w, degree)
    disk_mask = rotate_image(disk_mask, image_h, image_w, degree)
    image_h, image_w = pig_mask.shape
    # image_w_new = 1600
    # image_h_new = int(image_h * 1600 / image_w)
    # pig_mask = cv2.resize(pig_mask, (image_w_new, image_h_new))
    # disk_mask = cv2.resize(disk_mask, (image_w_new, image_h_new))
    # image_h, image_w = image_h_new, image_w_new

    pig_cnt = get_contour(pig_mask)
    disk_cnt = get_contour(disk_mask)
    pig_area = cv2.contourArea(pig_cnt)
    disk_area = cv2.contourArea(disk_cnt)
    ##########

    pig_c_x, pig_c_y = get_contour_centroid(pig_cnt)
    disk_c_x, disk_c_y = get_contour_centroid(disk_cnt)

    pig_points = get_convex_hull_points(pig_cnt)
    disk_points = get_convex_hull_points(disk_cnt)

    ##### 判断猪脚朝向 如果猪脚超上则翻转图片 #####
    up_num = np.sum(pig_cnt[:, 1] < pig_c_y)
    down_num = np.sum(pig_cnt[:, 1] > pig_c_y)
    if up_num > down_num:
        pig_c_y = image_h - 1 - pig_c_y
        pig_cnt[:, 1] = image_h - 1 - pig_cnt[:, 1]
        pig_points = get_convex_hull_points(pig_cnt)

        disk_c_y = image_h - 1 - disk_c_y
        disk_cnt[:, 1] = image_h - 1 - disk_cnt[:, 1]
        disk_points = get_convex_hull_points(disk_cnt)
    ##########

    return pig_cnt, pig_area, pig_c_x, pig_c_y, pig_points, disk_cnt, disk_area, disk_c_x, disk_c_y, disk_points, image_h, image_w, degree, pig_color_graylevel, pig_color_brightness, pig_cnt0


def divide_points(points):
    """若凸包点上下差距过大，分为上下两个点集

    :param points:
    :return:
    """
    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])
    mid = (y_max + y_min) / 2
    up = []
    down = []
    for i in range(points.shape[0]):
        if points[i, 1] < mid:
            up.append(np.expand_dims(points[i, :], 0))
        else:
            down.append(np.expand_dims(points[i, :], 0))
    up = np.concatenate(up, axis=0)
    down = np.concatenate(down, axis=0)
    return up, down


def distance_btw_points(points1, points2):
    """计算两个点集之间的最大距离

    :param points1:
    :param points2:
    :return:
    """
    points1_points2T = np.dot(points1, np.transpose(points2))
    points1_sq = np.tile(np.sum(points1 ** 2, axis=1, keepdims=True), (1, points1_points2T.shape[1]))
    points2_sq = np.tile(np.sum(np.transpose(points2) ** 2, axis=0, keepdims=True), (points1_points2T.shape[0], 1))
    dist = np.sqrt(points1_sq+points2_sq-2*points1_points2T)
    length = np.amax(dist)
    i_, j_ = np.unravel_index(dist.argmax(), dist.shape)

    ##### Plots for debugging #####
    if plot_flag:
        x1, y1 = points1[i_, :]
        x2, y2 = points2[j_, :]
        plt.plot([x1, x2], [y1, y2])
    ##########

    return length, points1[i_, :], points2[j_, :]


def get_pig_length(convex_points):
    """获取猪体长(长轴长度) length of pig long axis

    :param convex_points:
    :return:
    """

    left = np.min(convex_points[:, 0])
    right = np.max(convex_points[:, 0])
    thread = (right - left) * 0.1
    left_points = []
    right_points = []
    for i in range(convex_points.shape[0]):
        if convex_points[i, 0] < left + thread:
            left_points.append(np.expand_dims(convex_points[i, :], 0))
        elif convex_points[i, 0] > right - thread:
            right_points.append(np.expand_dims(convex_points[i, :], 0))
    left_points = np.concatenate(left_points, axis=0)

    right_points = np.concatenate(right_points, axis=0)

    range_left = np.max(left_points[:, 1]) - np.min(left_points[:, 1])
    range_right = np.max(right_points[:, 1]) - np.min(right_points[:, 1])

    range_thres = 450  # 750
    if range_left < range_thres:
        if range_right < range_thres:
            pig_length = distance_btw_points(left_points, right_points)
        else:
            right_up, right_down = divide_points(right_points)
            pig_length = distance_btw_points(left_points, right_up)

    else:
        left_up, left_down = divide_points(left_points)
        if range_right < range_thres:
            pig_length = distance_btw_points(left_up, right_points)

        else:
            right_up, right_down = divide_points(right_points)
            pig_length = distance_btw_points(left_up, right_up)

    return pig_length


def get_pig_length2(cnt, pig_c_x, pig_c_y, image_w, image_h):
    """获取猪尺寸(长、短轴长度) length along principal components of convex points

    :param:
    :return:
    """
    _, inside_points = get_points_inside_contour(cnt, image_h, image_w)

    y0 = round(pig_c_y)
    idices = np.where(inside_points[:, 1] == y0)
    axis_points = np.squeeze(inside_points[idices, :])
    left = np.min(axis_points[:, 0])
    right = np.max(axis_points[:, 0])
    pig_long_axis = right-left

    x0 = round(pig_c_x)
    idices = np.where(inside_points[:, 0] == x0)
    axis_points = np.squeeze(inside_points[idices, :])
    up = np.min(axis_points[:, 1])
    down = np.max(axis_points[:, 1])
    pig_short_axis = down - up
    ##### Real short axis #####
    axis_points = axis_points[:, 1].tolist()
    axis_ref = range(up, down+1)
    axis_blank = list(set(axis_ref)-set(axis_points))
    if axis_blank != []:
        down = min(axis_blank)-1
    #####
    pig_short_axis_2 = down-up

    ##### Plots for debugging #####
    if plot_flag:
        plt.plot([left, right], [y0, y0])
        plt.plot([x0, x0], [up, down])
    ##########

    return pig_long_axis, pig_short_axis, pig_short_axis_2


def pig_measure(pig_cnt, pig_c_x, pig_c_y, convex_points, image_w, image_h):
    """测量猪的 long axis length 和 centroid

    :param pig_c_x:
    :param pig_c_y:
    :param convex_points:
    :param img:
    :param image_w:
    :param image_h:
    :return:
    """
    convex_m = cv2.moments(convex_points)
    pig_hull_c_x = int(convex_m['m10'] / convex_m['m00'])
    pig_hull_c_y = int(convex_m['m01'] / convex_m['m00'])

    ##### Plots for debugging #####
    if plot_flag:
        plt.plot(pig_hull_c_x, pig_hull_c_y, 'X', color='black')
    ##########

    # pig_length = get_pig_length(convex_points)
    pig_length, pig_short_axis, pig_thickness = get_pig_length2(pig_cnt, pig_c_x, pig_c_y, image_w, image_h)

    pig_centroid = np.sqrt(
        np.sum(np.square(np.array([pig_c_x, pig_c_y]) - np.array([image_w / 2, image_h / 2]))))

    pig_dist_cnt_hull = np.sqrt(np.square(pig_hull_c_x-pig_c_x) + np.square(pig_hull_c_y-pig_c_y))

    # return pig_length, pig_centroid, pig_dist_cnt_hull
    return pig_length, pig_short_axis, pig_thickness, pig_centroid, pig_dist_cnt_hull


def disk_measure(disk_cnt, disk_c_x, disk_c_y, disk_convex, image_w, image_h):
    """测量盘的 length, height 和 centroid


    :param disk_c_x:
    :param disk_c_y:
    :param disk_convex:
    :param image_w:
    :param image_h:
    :return:
    """

    left = []
    right = []
    for i in range(disk_convex.shape[0]):
        if disk_convex[i, 0] < disk_c_x:
            left.append(np.expand_dims(disk_convex[i, :], 0))
        else:
            right.append(np.expand_dims(disk_convex[i, :], 0))

    left = np.concatenate(left, 0)
    right = np.concatenate(right, 0)
    disk_length, point1, point2 = distance_btw_points(left, right)

    disk_centroid = np.sqrt(
        np.sum(np.square(np.array([disk_c_x, disk_c_y]) - np.array([image_w / 2, image_h / 2]))))

    # ##### BAAAAAD #####
    #
    # grad = (point2[1]-point1[1]+1e-7)/(point2[0]-point1[0]+1e-7)
    # _, inside_points = get_points_inside_contour(disk_cnt, image_h, image_w)
    #
    # axis_points = np.empty(shape=[0, 2])
    # for k in range(inside_points.shape[0]):
    #     grad2 = (inside_points[k, 1]-disk_c_y+1e-7)/(inside_points[k, 0]-disk_c_x+1e-7)
    #     if abs(grad*grad2+1) < 0.01:
    #         axis_points = np.append(axis_points, [inside_points[k]], axis=0)
    # # print(axis_points)
    # if axis_points.shape[0] < 2:
    #     if grad==0:
    #         y1_ = np.amin(inside_points[:, 1])
    #         y2_ = np.amax(inside_points[:, 1])
    #         point1_ = np.array([disk_c_x, y1_])
    #         point2_ = np.array([disk_c_x, y2_])
    #     else:
    #         x1_ = np.amin(inside_points[:, 0])
    #         x2_ = np.amin(inside_points[:, 0])
    #         point1_ = np.array([x1_, disk_c_y])
    #         point2_ = np.array([x2_, disk_c_y])
    # else:
    #     idx1_ = np.argmin(axis_points[:, 1])
    #     idx2_ = np.argmax(axis_points[:, 1])
    #     point1_ = axis_points[idx1_, :]
    #     point2_ = axis_points[idx2_, :]
    # disk_short_axis = np.sqrt(np.sum(np.square(point1_-point2_)))

    # ##### Plots for debugging #####
    # if plot_flag:
    #     # print(point1_)
    #     # print(point2_)
    #     plt.plot([point1_[0], point2_[0]], [point1_[1], point2_[1]])
    # ##########

    #####

    # return disk_length, disk_short_axis, disk_centroid
    return disk_length, disk_centroid


def disk_is_inside(pig_cnt, disk_cnt, h, w):
    pig_mask, _ = get_points_inside_contour(pig_cnt, h, w)
    disk_mask, _ = get_points_inside_contour(disk_cnt, h, w)
    pig_flat = np.reshape(pig_mask[..., 0], [-1, 1])
    disk_flat = np.reshape(disk_mask[..., 0], [-1, 1])
    inter = np.sum(np.multiply(pig_flat, disk_flat))
    disk_area = np.sum(disk_flat)
    ratio = inter/disk_area
    if ratio > 0.9:
        return 1
    else:
        return 0

def get_features_rows(label_name, opt):  ## multiprocessing usage
    global idx
    idx += 1
    label = label_name.split('/')[-1]
    print('Processing example {}: {}'.format(idx, label))

    real_pig_length = label.split('_')[1].replace('cm', '')
    real_pig_weight = label.split('_')[3].replace('kg', '')

    # start_time = time.time()
    pig_mask_gray = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
    disk_mask_gray = cv2.imread(label_name.replace('pig_mask', 'disk_mask'), cv2.IMREAD_GRAYSCALE)
    raw_img = cv2.imread(label_name.replace('pig_mask', 'raw'))

    pig_mask = cv2.threshold(pig_mask_gray, 127, 255, cv2.THRESH_BINARY)[1]
    disk_mask = cv2.threshold(disk_mask_gray, 127, 255, cv2.THRESH_BINARY)[1]

    # 轮廓，重心X坐标，重心Y坐标，凸包点集
    pig_cnt, pig_area, pig_c_x, pig_c_y, pig_convex, disk_cnt, disk_area, \
    disk_c_x, disk_c_y, disk_convex, image_h, image_w, angle, pig_color_gray, pig_color_br, pig_cnt0 = \
        image_preprocessing(pig_mask, disk_mask, raw_img)

    pig_length, pig_short_axis, pig_thickness, pig_center, distance_pig_cnt_hull = \
        pig_measure(pig_cnt, pig_c_x, pig_c_y, pig_convex, image_w, image_h)

    # disk_length, disk_short_axis, disk_center = \
    #     disk_measure(disk_cnt, disk_c_x, disk_c_y, disk_convex, image_w, image_h)
    disk_length, disk_center = \
        disk_measure(disk_cnt, disk_c_x, disk_c_y, disk_convex, image_w, image_h)

    dist_centroids_x = pig_c_x - disk_c_x
    dist_centroids_y = pig_c_y - disk_c_y
    long_axis_ratio = pig_length / disk_length

    pig_area = cv2.contourArea(pig_cnt)
    pig_perimeter = cv2.arcLength(pig_cnt, True)
    area_ratio = pig_area / cv2.contourArea(disk_cnt)
    disk_inside = disk_is_inside(pig_cnt, disk_cnt, image_h, image_w)

    csv_row_l = {CSV_C0_NAME: label,
               CSV_C1_NAME: dist_centroids_x,
               CSV_C2_NAME: dist_centroids_y,
               CSV_C3_NAME: long_axis_ratio,
               CSV_C4_NAME: disk_length,
               CSV_C5_NAME: disk_center,
               CSV_C6_NAME: pig_center,
               CSV_C7_NAME: distance_pig_cnt_hull,
               CSV_C8_NAME: pig_area,
               CSV_C9_NAME: pig_perimeter,
               CSV_C10_NAME: pig_short_axis,
               CSV_C11_NAME: pig_thickness,
               CSV_C12_NAME: angle,
               # CSV_C13_NAME: disk_short_axis,
               CSV_C16_NAME: disk_inside,
               CSV_CF_NAME_L: real_pig_length}

    if real_pig_weight == '0':
        csv_row_w = None
    else:
        csv_row_w = {CSV_C0_NAME: label,
                   CSV_C1_NAME: dist_centroids_x,
                   CSV_C2_NAME: dist_centroids_y,
                   CSV_C3_NAME: long_axis_ratio,
                   CSV_C4_NAME: disk_length,
                   CSV_C5_NAME: disk_center,
                   CSV_C6_NAME: pig_center,
                   CSV_C7_NAME: distance_pig_cnt_hull,
                   CSV_C8_NAME: pig_area,
                   CSV_C9_NAME: pig_perimeter,
                   CSV_C10_NAME: pig_short_axis,
                   CSV_C11_NAME: pig_thickness,
                   CSV_C12_NAME: angle,
                   # CSV_C13_NAME: disk_short_axis,
                   CSV_C14_NAME: pig_color_gray,
                   CSV_C14_NAME_2: pig_color_br,
                   CSV_C15_NAME: area_ratio,
                   CSV_C16_NAME: disk_inside,
                   CSV_CF_NAME_W: real_pig_weight}

    # print("--- {}s seconds ---".format((time.time() - start_time)))

    ##### Generate outputs for debugging #####
    if plot_flag:
        try:
            os.mkdir(osp.join('regression_data', opt, 'output'))
        except:
            pass
        plt.plot(disk_cnt[:, 0], disk_cnt[:, 1])
        plt.plot(pig_cnt[:, 0], pig_cnt[:, 1])
        plt.plot(disk_c_x, disk_c_y, 'o')
        plt.plot(pig_c_x, pig_c_y, 'v')
        plt.xlim((0, image_w))
        plt.ylim((0, image_h))
        plt.gca().set_aspect('equal')
        plt.gca().invert_yaxis()
        plt.savefig(osp.join('regression_data', opt, 'output', label.replace('jpg', 'png')))
        plt.close()
    ##########

    return csv_row_l, csv_row_w


def extract(opt):
    label_path = osp.join('regression_data2', opt, 'pig_mask')
    label_names = [osp.join(label_path, label_name) for label_name in os.listdir(label_path)]

    global idx
    idx = 0

    print('============================== {} =============================='.format(opt))
    print('{} examples to process.'.format(len(label_names)))

    start_time_ = time.time()

    pool = ThreadPool(processes=100)
    rows = pool.starmap(get_features_rows,
                            zip(label_names, itertools.repeat(opt)))
    pool.close()
    pool.join()

    print("--- {}s seconds in total ---".format((time.time() - start_time_)))

    ##### write #####

    if write_flag:

        fieldnames_l = [CSV_C0_NAME,
                        CSV_C1_NAME,
                        CSV_C2_NAME,
                        CSV_C3_NAME,
                        CSV_C4_NAME,
                        CSV_C5_NAME,
                        CSV_C6_NAME,
                        CSV_C7_NAME,
                        CSV_C8_NAME,
                        CSV_C9_NAME,
                        CSV_C10_NAME,
                        CSV_C11_NAME,
                        CSV_C12_NAME,
                        # CSV_C13_NAME,
                        CSV_C16_NAME,
                        CSV_CF_NAME_L]

        fieldnames_w = [CSV_C0_NAME,
                        CSV_C1_NAME,
                        CSV_C2_NAME,
                        CSV_C3_NAME,
                        CSV_C4_NAME,
                        CSV_C5_NAME,
                        CSV_C6_NAME,
                        CSV_C7_NAME,
                        CSV_C8_NAME,
                        CSV_C9_NAME,
                        CSV_C10_NAME,
                        CSV_C11_NAME,
                        CSV_C12_NAME,
                        # CSV_C13_NAME,
                        CSV_C14_NAME,
                        CSV_C14_NAME_2,
                        CSV_C15_NAME,
                        CSV_C16_NAME,
                        CSV_CF_NAME_W]

        with open('length_'+opt+'.csv', 'w') as csvfile_l, open('weight_'+opt+'.csv', 'w') as csvfile_w:
            writer_l = csv.DictWriter(csvfile_l, fieldnames=fieldnames_l)
            writer_l.writeheader()
            writer_w = csv.DictWriter(csvfile_w, fieldnames=fieldnames_w)
            writer_w.writeheader()
            for row in rows:
                l_row, w_row = row

                if l_row is not None:
                    writer_l.writerow({
                        CSV_C0_NAME: l_row[CSV_C0_NAME],
                        CSV_C1_NAME: l_row[CSV_C1_NAME],
                        CSV_C2_NAME: l_row[CSV_C2_NAME],
                        CSV_C3_NAME: l_row[CSV_C3_NAME],
                        CSV_C4_NAME: l_row[CSV_C4_NAME],
                        CSV_C5_NAME: l_row[CSV_C5_NAME],
                        CSV_C6_NAME: l_row[CSV_C6_NAME],
                        CSV_C7_NAME: l_row[CSV_C7_NAME],
                        CSV_C8_NAME: l_row[CSV_C8_NAME],
                        CSV_C9_NAME: l_row[CSV_C9_NAME],
                        CSV_C10_NAME: l_row[CSV_C10_NAME],
                        CSV_C11_NAME: l_row[CSV_C11_NAME],
                        CSV_C12_NAME: l_row[CSV_C12_NAME],
                        # CSV_C13_NAME: l_row[CSV_C13_NAME],
                        CSV_C16_NAME: l_row[CSV_C16_NAME],
                        CSV_CF_NAME_L: l_row[CSV_CF_NAME_L]
                    })

                if w_row is not None:
                    writer_w.writerow({
                        CSV_C0_NAME: w_row[CSV_C0_NAME],
                        CSV_C1_NAME: w_row[CSV_C1_NAME],
                        CSV_C2_NAME: w_row[CSV_C2_NAME],
                        CSV_C3_NAME: w_row[CSV_C3_NAME],
                        CSV_C4_NAME: w_row[CSV_C4_NAME],
                        CSV_C5_NAME: w_row[CSV_C5_NAME],
                        CSV_C6_NAME: w_row[CSV_C6_NAME],
                        CSV_C7_NAME: w_row[CSV_C7_NAME],
                        CSV_C8_NAME: w_row[CSV_C8_NAME],
                        CSV_C9_NAME: w_row[CSV_C9_NAME],
                        CSV_C10_NAME: w_row[CSV_C10_NAME],
                        CSV_C11_NAME: w_row[CSV_C11_NAME],
                        CSV_C12_NAME: w_row[CSV_C12_NAME],
                        # CSV_C13_NAME: w_row[CSV_C13_NAME],
                        CSV_C14_NAME: w_row[CSV_C14_NAME],
                        CSV_C14_NAME_2: w_row[CSV_C14_NAME_2],
                        CSV_C15_NAME: w_row[CSV_C15_NAME],
                        CSV_C16_NAME: w_row[CSV_C16_NAME],
                        CSV_CF_NAME_W: w_row[CSV_CF_NAME_W]
                    })

if __name__=='__main__':
    CSV_C0_NAME = 'file_name'
    CSV_C1_NAME = 'dist_centroids_x'
    CSV_C2_NAME = 'dist_centroids_y'
    CSV_C3_NAME = 'length_ratio'
    CSV_C4_NAME = 'disk_length'
    CSV_C5_NAME = 'disk_center'  # distance between centroid of disk and center of the image
    CSV_C6_NAME = 'pig_center'  # distance between centroid of pig and center of the image
    CSV_C7_NAME = 'pig_hull_center'  # distance between centroid of pig and centroid of its convex hull
    CSV_C8_NAME = 'pig_area'
    CSV_C9_NAME = 'pig_perimeter'
    CSV_C10_NAME = 'pig_short_axis'
    CSV_C11_NAME = 'pig_thickness'
    CSV_C12_NAME = 'angle'
    CSV_C13_NAME = 'disk_short_axis'
    CSV_C14_NAME = 'pig_color_gray'
    CSV_C14_NAME_2 = 'pig_color_br'
    CSV_C15_NAME = 'area_ratio'
    CSV_C16_NAME = 'disk_is_inside'
    CSV_CF_NAME_L = 'real_pig_length'
    CSV_CF_NAME_W = 'real_pig_weight'

    idx = 0
    extract('train')
    extract('test')