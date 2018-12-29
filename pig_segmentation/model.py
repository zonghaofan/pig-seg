#coding:utf-8
import tensorflow as tf
import vgg16
import cv2
import numpy as np

#图片尺寸 704，输出的预测结果为图片尺寸的一半大小
img_size = 704
label_size = img_size / 2


class Model:
    def __init__(self):
        """Load vgg16 pre-trained model

            Args:
                None

            Returns:
                None
            """
        self.vgg = vgg16.Vgg16()

        self.input_holder = tf.placeholder(tf.float32, [None, img_size, img_size, 3],name='input_placeholder')
        self.label_holder = tf.placeholder(tf.float32, [None,label_size*label_size, 2],name='labels_placeholder')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        self.sobel_fx, self.sobel_fy = self.sobel_filter()

        self.contour_th = 1.5
        self.contour_weight = 0.0001

    def build_model(self):
        """Build a segmentation architecture

            Args:
                None

            Returns:
                None
            """
        batch_size = tf.shape(self.input_holder)[0]
        #build the VGG-16 model
        vgg = self.vgg
        vgg.build(self.input_holder)

        fea_dim = 32
        # self.conv6_1 = tf.nn.relu(self.Conv_2d(vgg.pool5, [3, 3, 512, 1024], 0.01, padding='SAME', name='conv6_1'))
        self.conv6_1 = tf.nn.relu(self.Conv_2d(vgg.pool5,[3, 3, 512, 1024], padding='SAME', name='conv6_1', training=self.is_training))
        # self.conv6_2 = tf.nn.relu(self.Conv_2d(self.conv6_1,[3, 3, 1024, 1024], 0.01, padding='SAME', name='conv6_2'))
        self.conv6_2 = tf.nn.relu(self.Conv_2d(self.conv6_1, [3, 3, 1024, 1024], padding='SAME', name='conv6_2', training=self.is_training))
        # self.conv6_3 = tf.nn.relu(self.Conv_2d(self.conv6_2,[3, 3, 1024, 1024], 0.01, padding='SAME', name='conv6_3'))
        self.conv6_3 = tf.nn.relu(self.Conv_2d(self.conv6_2, [3, 3, 1024, 1024], padding='SAME', name='conv6_3', training=self.is_training))
        self.pool6 = tf.nn.max_pool(self.conv6_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name='pool6')

        '''self.conv7_1 = tf.nn.relu(self.Conv_2d(self.pool6, [3, 3, 1024, 1024], 0.01, padding='SAME', name='conv7_1'))
        self.conv7_2 = tf.nn.relu(self.Conv_2d(self.conv7_1, [3, 3, 1024, 1024], 0.01, padding='SAME', name='conv7_2'))
        self.conv7_3 = tf.nn.relu(self.Conv_2d(self.conv7_2, [3, 3, 1024, 1024], 0.01, padding='SAME', name='conv7_3'))
        self.pool7 = tf.nn.max_pool(self.conv7_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool7')

        self.conv8_1 = tf.nn.relu(self.Conv_2d(self.pool7, [3, 3, 1024, 1024], 0.01, padding='SAME', name='conv8_1'))
        self.conv8_2 = tf.nn.relu(self.Conv_2d(self.conv8_1, [3, 3, 1024, 1024], 0.01, padding='SAME', name='conv8_2'))
        self.conv8_3 = tf.nn.relu(self.Conv_2d(self.conv8_2, [3, 3, 1024, 1024], 0.01, padding='SAME', name='conv8_3'))
        self.pool8 = tf.nn.max_pool(self.conv8_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME', name='pool8')'''

        #Global Feature and Global Score

        self.Fea_Global_1 = tf.nn.relu(self.Conv_2d(vgg.pool5, [5, 5, 512, fea_dim],
                                                    padding='VALID', name='Fea_Global_1', training=self.is_training))
        self.Fea_Global_2 = tf.nn.relu(self.Conv_2d(self.Fea_Global_1, [5, 5, fea_dim, fea_dim],
                                                    padding='VALID', name='Fea_Global_2', training=self.is_training))
        self.Fea_Global = self.Conv_2d(self.Fea_Global_2, [3, 3, fea_dim, fea_dim],
                                       padding='VALID', name='Fea_Global', training=self.is_training)

        #Local Score
        #self.Fea_P8 = tf.nn.relu(self.Conv_2d(self.pool8, [3, 3, 1024, fea_dim], 0.01, padding='SAME', name='Fea_P8'))
        #self.Fea_P7 = tf.nn.relu(self.Conv_2d(self.pool7, [3, 3, 1024, fea_dim], 0.01, padding='SAME', name='Fea_P7'))
        self.Fea_P6 = tf.nn.relu(self.Conv_2d(self.pool6, [3, 3, 1024, fea_dim],  padding='SAME', name='Fea_P6', training=self.is_training))
        self.Fea_P5 = tf.nn.relu(self.Conv_2d(vgg.pool5, [3, 3, 512, fea_dim],  padding='SAME', name='Fea_P5', training=self.is_training))
        self.Fea_P4 = tf.nn.relu(self.Conv_2d(vgg.pool4, [3, 3, 512, fea_dim],  padding='SAME', name='Fea_P4', training=self.is_training))
        self.Fea_P3 = tf.nn.relu(self.Conv_2d(vgg.pool3, [3, 3, 256, fea_dim],  padding='SAME', name='Fea_P3', training=self.is_training))
        self.Fea_P2 = tf.nn.relu(self.Conv_2d(vgg.pool2, [3, 3, 128, fea_dim], padding='SAME', name='Fea_P2', training=self.is_training))
        self.Fea_P1 = tf.nn.relu(self.Conv_2d(vgg.pool1, [3, 3, 64, fea_dim], padding='SAME', name='Fea_P1', training=self.is_training))


        #self.Fea_P8_LC = self.Contrast_Layer(self.Fea_P8, 3)
        #self.Fea_P7_LC = self.Contrast_Layer(self.Fea_P7, 3)
        self.Fea_P6_LC = self.Contrast_Layer(self.Fea_P6, 3)
        self.Fea_P5_LC = self.Contrast_Layer(self.Fea_P5, 3)
        self.Fea_P4_LC = self.Contrast_Layer(self.Fea_P4, 3)
        self.Fea_P3_LC = self.Contrast_Layer(self.Fea_P3, 3)
        self.Fea_P2_LC = self.Contrast_Layer(self.Fea_P2, 3)
        self.Fea_P1_LC = self.Contrast_Layer(self.Fea_P1, 3)

        #Deconv Layer
        #self.Fea_P8_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P8, self.Fea_P8_LC], axis=3),
         #                                          [1, 22, 22, fea_dim], 5, 2, name='Fea_P8_Deconv'))
        #self.Fea_P7_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P7, self.Fea_P7_LC], axis=3),
         #                                          [1, 22, 2, fea_dim], 5, 2, name='Fea_P7_Deconv'))
        self.Fea_P6_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P6, self.Fea_P6_LC], axis=3),
                                                   [batch_size, 22, 22, fea_dim*1], 5, 2,training=self.is_training,name='Fea_P6_Deconv'))
        self.Fea_P5_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P5, self.Fea_P5_LC], axis=3),
                                                   [batch_size, 44, 44, fea_dim*2], 5, 2,training=self.is_training,name='Fea_P5_Deconv'))
        self.Fea_P4_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P4, self.Fea_P4_LC, self.Fea_P5_Up], axis=3),
                                                   [batch_size, 88, 88, fea_dim*3], 5, 2,training=self.is_training,name='Fea_P4_Deconv'))
        self.Fea_P3_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P3, self.Fea_P3_LC, self.Fea_P4_Up], axis=3),
                                                   [batch_size, 176, 176, fea_dim*4], 5, 2,training=self.is_training,name='Fea_P3_Deconv'))
        self.Fea_P2_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P2, self.Fea_P2_LC, self.Fea_P3_Up], axis=3),
                                                   [batch_size, 352, 352, fea_dim*5], 5, 2,training=self.is_training,name='Fea_P2_Deconv'))

        '''self.Fea_P5_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P5, self.Fea_P5_LC], axis=3),
                                                   [1, 22, 22, fea_dim * 1], 5, 2, name='Fea_P5_Deconv'))
        self.Fea_P4_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P4, self.Fea_P4_LC, self.Fea_P5_Up], axis=3),
                                                   [1, 44, 44, fea_dim * 2], 5, 2, name='Fea_P4_Deconv'))
        self.Fea_P3_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P3, self.Fea_P3_LC, self.Fea_P4_Up], axis=3),
                                                   [1, 88, 88, fea_dim * 3], 5, 2, name='Fea_P3_Deconv'))
        self.Fea_P2_Up = tf.nn.relu(self.Deconv_2d(tf.concat([self.Fea_P2, self.Fea_P2_LC, self.Fea_P3_Up], axis=3),
                                                   [1, 176, 176, fea_dim * 4], 5, 2, name='Fea_P2_Deconv'))'''

        self.Local_Fea = self.Conv_2d(tf.concat([self.Fea_P1, self.Fea_P1_LC, self.Fea_P2_Up], axis=3),
                                      [1, 1, fea_dim*7, fea_dim*5], padding='SAME', name='Local_Fea', training=self.is_training)
        self.Local_Score = self.Conv_2d(self.Local_Fea, [1, 1, fea_dim*5, 2],  padding='SAME', name='Local_Score', training=self.is_training)

        self.Global_Score = self.Conv_2d(self.Fea_Global,
                                         [1, 1, fea_dim, 2],  padding='SAME', name='Global_Score', training=self.is_training)

        #self.Score = self.Local_Score + self.Global_Score
        self.Score = self.Local_Score
        self.Score = tf.reshape(self.Score, [-1,label_size*label_size,2])

        self.Prob = tf.nn.softmax(self.Score,name='softmax')

        #Get the contour term
        self.Prob_C = tf.reshape(self.Prob, [-1, label_size, label_size, 2])
        self.Prob_Grad = tf.tanh(self.im_gradient(self.Prob_C))
        self.Prob_Grad = tf.tanh(tf.reduce_sum(self.im_gradient(self.Prob_C), reduction_indices=3, keep_dims=True))

        self.label_C = tf.reshape(self.label_holder, [-1, label_size, label_size, 2])
        self.label_Grad = tf.cast(tf.greater(self.im_gradient(self.label_C), self.contour_th), tf.float32)
        self.label_Grad = tf.cast(tf.greater(tf.reduce_sum(self.im_gradient(self.label_C),
                                                           reduction_indices=3, keep_dims=True),
                                             self.contour_th), tf.float32)

        self.C_IoU_LOSS = self.Loss_IoU(self.Prob_Grad, self.label_Grad)

        # self.Contour_Loss = self.Loss_Contour(self.Prob_Grad, self.label_Grad)

        self.CE_Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Score,
                                                                              labels=self.label_holder))
        self.Loss_total = self.C_IoU_LOSS + self.CE_Loss

        self.pred_flat=tf.reshape(self.Prob_C[...,0],[-1,label_size*label_size])
        self.true_flat=tf.reshape(self.label_holder[...,0], [-1, label_size*label_size])
        self.TP =  self.true_flat*self.pred_flat
        self.FP=(1.-self.true_flat)*self.pred_flat
        self.FN = self.true_flat *(1.-self.pred_flat)
        self.precision = tf.reduce_sum(self.TP) / (tf.reduce_sum(self.TP)+tf.reduce_sum(self.FP) + 1e-7)
        self.recall = tf.reduce_sum(self.TP) / (tf.reduce_sum(self.TP)+tf.reduce_sum(self.FN) + 1e-7)
        self.f1_score = 2 / (1 / self.precision + 1 / self.recall + 1e-7)

        # self.correct_prediction = tf.equal(tf.argmax(self.Score,axis=2), tf.argmax(self.label_holder, 2))
        # self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


    def Conv_2d(self, input_, shape,name, training=False,padding='SAME'):
        """Build a conv2d function

            Args:
                input_ (4-D Tensor): (N, H, W, C)
                shape : (N,H,W,C)
                stddev :
                name :
                padding : default is Same

            Returns:
                conv (4-D Tensor): (N, H, W, C)
                Same shape as the `input` tensor
            """
        with tf.variable_scope(name):
            if shape[3]>=shape[2]:
                W = tf.get_variable('W',
                                    shape=[shape[0],shape[1],shape[2],shape[3]/shape[2]],
                                    initializer=tf.contrib.layers.xavier_initializer())

                conv=tf.nn.depthwise_conv2d(input_, W, [1, 1, 1, 1], padding=padding)
            else:
                W = tf.get_variable('W',
                                    shape,
                                    initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding=padding)
            b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
            conv = tf.nn.bias_add(conv, b)

            conv_bn=tf.layers.batch_normalization(conv, training=training,name='conv_bn')

            return conv_bn

    # def Conv_2d_batch_norm(self, input_, shape, stddev, name, padding='SAME', training=True):
    #     conv = self.Conv_2d(input_, shape, stddev, name, padding)
    #     conv_bn = tf.layers.batch_normalization(conv, training=training)
    #     return conv_bn


    def Deconv_2d(self, input_, output_shape,
                  k_s=3, st_s=2,padding='SAME', training=False,name="deconv2d"):
        """Build a deconv2d function

            Args:
                input_ (4-D Tensor): (N, H, W, C)
                output_shape : (N,H,W,C)
                k_s :
                st_s :
                stddev :
                name :
                padding : default is SAME

            Returns:
                deconv (4-D Tensor): (N, H, W, C)
                Same shape as the `input` tensor
            """
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                                shape=[k_s, k_s, output_shape[3], input_.get_shape()[3]],
                                initializer=tf.contrib.layers.xavier_initializer())

            deconv = tf.nn.conv2d_transpose(input_, W, output_shape=output_shape,
                                            strides=[1, st_s, st_s, 1], padding=padding)

            b = tf.get_variable('b', [output_shape[3]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, b)

            deconv_bn=tf.layers.batch_normalization(deconv, training=training,name='deconv_bn')

        return deconv_bn

    def Contrast_Layer(self, input_, k_s=3):
        """Build a subtract function

            Args:
                input_ (4-D Tensor): (N, H, W, C)
                k_s :

            Returns:
                output (4-D Tensor): (N, H, W, C)
                Same shape as the `input` tensor
            """
        h_s = k_s / 2
        return tf.subtract(input_, tf.nn.avg_pool(tf.pad(input_, [[0, 0], [h_s, h_s], [h_s, h_s], [0, 0]], 'SYMMETRIC'),
                                                  ksize=[1, k_s, k_s, 1], strides=[1, 1, 1, 1], padding='VALID'))

    def sobel_filter(self):
        fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
        fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)

        fx = np.stack((fx, fx), axis=2)
        fy = np.stack((fy, fy), axis=2)

        fx = np.reshape(fx, (3, 3, 2, 1))
        fy = np.reshape(fy, (3, 3, 2, 1))
        #(3,3,2,1)
        tf_fx = tf.Variable(tf.constant(fx))
        tf_fy = tf.Variable(tf.constant(fy))

        return tf_fx, tf_fy

    def im_gradient(self, im):
        gx = tf.nn.depthwise_conv2d(tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC'),
                                    self.sobel_fx, [1, 1, 1, 1], padding='VALID')
        gy = tf.nn.depthwise_conv2d(tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC'),
                                    self.sobel_fy, [1, 1, 1, 1], padding='VALID')
        return tf.sqrt(tf.add(tf.square(gx), tf.square(gy)))

    def Loss_IoU(self, pred, gt):
        inter = tf.reduce_sum(tf.multiply(pred, gt))
        union = tf.add(tf.reduce_sum(tf.square(pred)), tf.reduce_sum(tf.square(gt)))

        if inter == 0:
            return 0
        else:
            return 1 - 2*(inter+1)/(union + 1)

    # def Loss_IoU(self, pred, gt):
    #     inter = tf.reduce_sum(tf.multiply(pred, gt))
    #     union = tf.reduce_sum(tf.minimum(tf.subtract(tf.add(pred, gt), tf.multiply(pred, gt)), 1))
    #     # union = tf.reduce_sum(tf.subtract(tf.add(pred, gt), tf.multiply(pred, gt)))
    #
    #     loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.div(inter+1e-7, union+1e-7))
    #     return loss


    def Loss_Contour(self, pred, gt):
        return tf.reduce_mean(-gt*tf.log(pred+0.00001) - (1-gt)*tf.log(1-pred+0.00001))

    def L2(self, tensor, wd=0.0005):
        return tf.mul(tf.nn.l2_loss(tensor), wd, name='L2-Loss')

