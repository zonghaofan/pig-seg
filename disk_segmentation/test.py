# coding:utf-8

import os
import cv2
import glob
import tensorflow as tf
import numpy as np
import argparse
from math import *
import time

h_out = 1200
w_out = 1600
h_in = 768 # 704 768
w_in = 768 # 704 1024
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
input_path = './test_data/shandong2_song_raw'
output_path = './test_data/shandong2_song_pred'

if not os.path.exists(output_path):
    os.makedirs(output_path)

def load_model():
    # file_meta = './model_v0/model.ckpt.meta'
    # file_ckpt = './model_v0/model.ckpt'
    file_meta = './model_v6_50/model_final.ckpt.meta'
    file_ckpt = './model_v6_50/model_final.ckpt'
    saver = tf.train.import_meta_graph(file_meta)

    sess = tf.InteractiveSession()
    saver.restore(sess, file_ckpt)
    return sess


def images_path(input_path):
    images_path_list = [os.path.join(input_path, i) for i in os.listdir(input_path)]
    return images_path_list


def read_image(image_path, gray=False):
    if gray:
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rotate_image(img, height, width):
    degree = 90
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    img_rotated = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    return img_rotated


def main():
    sess = load_model()
    X, mode = tf.get_collection('inputs')
    pred = tf.get_collection('outputs')[0]
    # pred = tf.get_collection('pred')[0]

    images_path_list = images_path(input_path)
    print('=============== {} test examples ==============='.format(len(images_path_list)))
    for i, path in enumerate(images_path_list):
        # print('==================== Test =====================')
        image = read_image(path)
        # print(image.shape)
        h_, w_ = image.shape[:2]
        if h_ > w_:
            image = rotate_image(image, h_, w_)
        image = cv2.resize(image, (w_in, h_in))
        print('Test example {}'.format(i+1))
        print(path.split('/')[3])
        start_time = time.time()
        label_pred = sess.run(pred, feed_dict={X: np.expand_dims(image, 0), mode: False})
        print("--- {}s seconds ---".format((time.time() - start_time)))
        merged = np.squeeze(label_pred) * 255
        merged = cv2.resize(merged, (w_out, h_out))
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        cv2.imwrite(output_path+'/'+path.split('/')[3], merged)


if __name__ == '__main__':
    main()






