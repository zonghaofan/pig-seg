import cv2
import numpy as np
import model
import os
import sys
import tensorflow as tf
import time
import vgg16


#这个脚本是用来测试pb文件的
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

path = '/notebooks/huludao'


def crop(pig_image):
    shape = pig_image.shape
    rate = float(shape[0]) / float(shape[1])
    if rate > 0.75:
        width = int(float(shape[1]) / 4 * 3)
        star = (shape[0] - width) / 2
        end = (shape[0] + width) / 2
        pig_image = pig_image[star:end, ...]
    elif rate < 0.75:
        height = int(float(shape[0]) / 3 * 4)
        star = (shape[1] - height) / 2
        end = (shape[1] + height) / 2
        pig_image = pig_image[:, star:end, :]

    return pig_image


if __name__ == "__main__":

    img_size = model.img_size
    label_size = model.label_size


    '''ckpt = tf.train.get_checkpoint_state('./model_bis/')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)'''

    with tf.gfile.FastGFile('./Model_176/expert-graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:

            init = tf.global_variables_initializer()
            sess.run(init)
            #print(X)
            X = sess.graph.get_tensor_by_name("input:0")
            print X
            out_softmax = sess.graph.get_tensor_by_name("softmax:0")
            print out_softmax
            #pred = sess.graph.get_tensor_by_name("output:0")
            #print pred


            datasets = ['MSRA-B']


            img = os.listdir(path)
            print('img',img)
            for i in img:
                if i != '.DS_Store':
                    #print('i',i)
                    imgs = os.listdir(path + '/' + i)
                    for f_img in imgs:
                        print(f_img)
                        if f_img[-4:] == '.jpg' or f_img[-5:] == '.jpeg':
                            print('f_img',f_img)

                            img = cv2.imread(path + '/' + i + '/' + f_img)
                            print('before crop',img.shape)

                            img = crop(img)
                            print('after crop',img.shape)

                            img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
                            img = img.reshape((1, img_size, img_size, 3))
                            print(img.shape)

                            start_time = time.time()
                            result = sess.run(out_softmax,feed_dict={X:img})
                            print("--- %s seconds ---" % (time.time() - start_time))
                            print('shape',result.shape)

                            result = np.reshape(result, (label_size, label_size, 2))
                            result = result[:, :, 0]*255
                            #_, result = cv2.threshold(result, 50, 255, cv2.THRESH_BINARY)

                            #result = cv2.resize(np.squeeze(result), (352, 1200))

                            try:
                                os.mkdir('/notebooks/NLDF/huludao/' + i)
                            except Exception as e:
                                print(e)
                                #print('path existed')

                            cv2.imwrite('/notebooks/NLDF/huludao/' + i + '/' + f_img, result)

                            print('Ok')


