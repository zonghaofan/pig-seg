import cv2
import numpy as np
import os
import tensorflow as tf
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
input_path='./test1/'
output_path='./test1_out'
if not os.path.exists(output_path):
    os.makedirs(output_path)
if __name__ == "__main__":
    img_size = 704
    label_size = 704/2
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./Model_pig_BN/model.ckpt.meta')
        # saver.restore(sess,tf.train.latest_checkpoint('./Model_pig_BN'))
        saver.restore(sess, './Model_pig_BN/model.ckpt')
        X,mode = tf.get_collection('inputs')
        pred = tf.get_collection('pred')[0]

        images_path_list=[os.path.join(input_path,i) for i in os.listdir(input_path)]
        for i,image_path in enumerate(images_path_list):
            img=cv2.imread(image_path)
            # print(img)
            img = cv2.resize(img, (img_size, img_size))
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = img.reshape((1, img_size, img_size, 3))
            start_time = time.time()
            result = sess.run(pred,feed_dict={X:np.expand_dims(img,0),mode: False})
            print("--- {}seconds ---,{}image" .format((time.time() - start_time),i))
            print('shape',result.shape)
            result=np.expand_dims(result[:, :, :, 0], axis=-1)
            result = result[0,...]*255
            result = cv2.resize(result, (1600, 1200))
            cv2.imwrite(output_path+'/'+image_path.split('/')[2],result)

