#coding:utf-8
import os
import sys

import numpy as np
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]

# https://github.com/machrisaa/tensorflow-vgg

class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        if vgg16_npy_path is None:
            path = sys.modules[self.__class__.__module__].__file__
            # print path
            path = os.path.abspath(os.path.join(path, os.pardir))
            # print path
            path = os.path.join(path, "vgg16.npy")
            print(path)
            vgg16_npy_path = path

        self.data_dict = np.load(vgg16_npy_path).item()
        # print(np.array(self.data_dict['conv1_1']).shape)
        print("npy file loaded success")

    def build(self, input, train=False):
        """"notice"""
        input=input/127.5-1
        self.conv1_1 = self._conv_layer(input, "conv1_1")
        print('conv1_1.shape={}'.format(self.conv1_1.shape))

        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        print('conv1_2.shape={}'.format(self.conv1_2.shape))

        self.pool1 = self._max_pool(self.conv1_2, 'pool1')
        print('pool1.shape={}'.format(self.pool1.shape))

        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        print('conv2_1.shape={}'.format(self.conv2_1.shape))

        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        print('conv2_2.shape={}'.format(self.conv2_2.shape))

        self.pool2 = self._max_pool(self.conv2_2, 'pool2')
        print('pool2.shape={}'.format(self.pool2.shape))

        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        print('conv3_1.shape={}'.format(self.conv3_1.shape))

        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        print('conv3_2.shape={}'.format(self.conv3_2.shape))

        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        print('conv3_3.shape={}'.format(self.conv3_3.shape))

        self.pool3 = self._max_pool(self.conv3_3, 'pool3')
        print('pool3.shape={}'.format(self.pool3.shape))


        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        print('conv4_1.shape={}'.format(self.conv4_1.shape))

        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        print('conv4_2.shape={}'.format(self.conv4_2.shape))

        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        print('conv4_3.shape={}'.format(self.conv4_3.shape))
        self.pool4 = self._max_pool(self.conv4_3, 'pool4')
        print('pool4.shape={}'.format(self.pool4.shape))

        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        print('conv5_1.shape={}'.format(self.conv5_1.shape))
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        print('conv5_2.shape={}'.format(self.conv5_2.shape))
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")
        print('conv5_3.shape={}'.format(self.conv5_3.shape))
        self.pool5 = self._max_pool(self.conv5_3, 'pool5')
        print('pool5.shape={}'.format(self.pool5.shape))


    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name=name)

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def _fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):

        return tf.Variable(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name="weights")

    def L2(self, tensor, wd=0.001):
        return tf.mul(tf.nn.l2_loss(tensor), wd, name='L2-Loss')

if __name__ == '__main__':
    vgg=Vgg16()
    input_holder = tf.placeholder(tf.float32, [1, 704, 704, 3], name='input')
    vgg.build(input_holder)
