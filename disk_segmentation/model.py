#coding:utf-8

import os
import tensorflow as tf
import numpy as np


def conv_conv_pool(x, n_filters, training, name, pool=True, activation=tf.nn.relu):
    with tf.variable_scope('layer{}'.format(name)):
        input = x
        for i, filters in enumerate(n_filters):
            conv = tf.layers.conv2d(input, filters, (3, 3), strides=1, padding='same', activation=None,
                                    name='conv_{}'.format(i+1), kernel_initializer=tf.variance_scaling_initializer())
            conv = tf.layers.batch_normalization(conv, training=training, name='bn_{}'.format(i+1))
            conv = activation(conv, name='relu{}_{}'.format(name, i+1))
            input = conv
        if pool is False:
            return conv
        pool = tf.layers.max_pooling2d(conv, pool_size=(2, 2), strides=2, name='pool_{}'.format(name))
        return conv, pool

def conv_conv_pool_depthwise(x, n_filters, training, name, pool=True, activation=tf.nn.relu):
    with tf.variable_scope('layer{}'.format(name)):
        in_filts = int(x.shape[3])
        filter = tf.get_variable('W1', shape=[3, 3, in_filts, n_filters[0]/in_filts], initializer=tf.variance_scaling_initializer())

        conv1 = tf.nn.depthwise_conv2d(x, filter, strides=[1, 1, 1, 1], padding='SAME', name='depthwise_conv_1')
        conv1 = tf.layers.batch_normalization(conv1, training=training, name='bn_1')
        conv1 = activation(conv1, name='relu{}_1'.format(name))

        conv2 = tf.layers.conv2d(conv1, n_filters[1], (3, 3), strides=1, padding='same', activation=None,
                                    name='conv_2', kernel_initializer=tf.variance_scaling_initializer())
        conv2 = tf.layers.batch_normalization(conv2, training=training, name='bn_2')
        conv2 = activation(conv2, name='relu{}_2'.format(name))

        if pool is False:
            return conv2
        pool = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=2, name='pool_{}'.format(name))
        return conv2, pool

def upsampling_2d(tensor, name, size=(2, 2)):
    h_, w_, c_ = tensor.get_shape().as_list()[1:]
    h_multi, w_multi = size
    h = h_multi * h_
    w = w_multi * w_
    target = tf.image.resize_nearest_neighbor(tensor, size=(h, w), name='upsample_{}'.format(name))

    return target


def upsampling_concat(input_A, input_B, name):
    upsampling = upsampling_2d(input_A, name=name, size=(2, 2))
    up_concat = tf.concat([upsampling, input_B], axis=-1, name='up_concat_{}'.format(name))

    return up_concat

def unet(input,training):
    #归一化[-1,1]
    input = input/127.5 - 1
    # r_g=tf.expand_dims(input[...,0]/(input[...,1]+1),-1)
    # r_b=tf.expand_dims(input[...,0]/(input[...,2]+1),-1)
    # g_b=tf.expand_dims(input[...,1]/(input[...,2]+1),-1)
    #
    # input = tf.concat([r_g,r_b,g_b],axis=-1)


    #input = tf.layers.conv2d(input,3,(1,1),name = 'color')   #filters:一个整数，输出空间的维度，也就是卷积核的数量
    conv1, pool1 = conv_conv_pool(input, [8, 8], training, name=1)       #卷积两次输出维度64维
    conv2, pool2 = conv_conv_pool(pool1, [16, 16], training, name=2)
    conv3, pool3 = conv_conv_pool(pool2, [32, 32], training, name=3)
    conv4, pool4 = conv_conv_pool(pool3, [64, 64], training, name=4)
    conv5 = conv_conv_pool(pool4, [128, 128], training, pool=False, name=5)

    up6 = upsampling_concat(conv5, conv4, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, pool=False, name=6)
    up7 = upsampling_concat(conv6, conv3, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, pool=False, name=7)
    up8 = upsampling_concat(conv7, conv2, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, pool=False, name=8)
    up9 = upsampling_concat(conv8, conv1, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, pool=False, name=9)

    return tf.layers.conv2d(conv9, 1, (1, 1), name='final', activation=None, padding='same')

def depthwise_unet(input, training):
    input = input / 127.5 - 1

    conv1, pool1 = conv_conv_pool(input, [8, 8], training, name=1)
    conv2, pool2 = conv_conv_pool_depthwise(pool1, [16, 16], training, name=2)
    conv3, pool3 = conv_conv_pool_depthwise(pool2, [32, 32], training, name=3)
    conv4, pool4 = conv_conv_pool_depthwise(pool3, [64, 64], training, name=4)
    conv5 = conv_conv_pool_depthwise(pool4, [128, 128], training, pool=False, name=5)

    up6 = upsampling_concat(conv5, conv4, name=6)
    conv6 = conv_conv_pool(up6, [64, 64], training, pool=False, name=6)
    up7 = upsampling_concat(conv6, conv3, name=7)
    conv7 = conv_conv_pool(up7, [32, 32], training, pool=False, name=7)
    up8 = upsampling_concat(conv7, conv2, name=8)
    conv8 = conv_conv_pool(up8, [16, 16], training, pool=False, name=8)
    up9 = upsampling_concat(conv8, conv1, name=9)
    conv9 = conv_conv_pool(up9, [8, 8], training, pool=False, name=9)

    return tf.layers.conv2d(conv9, 1, (1, 1), name='final', activation=None, padding='same')






