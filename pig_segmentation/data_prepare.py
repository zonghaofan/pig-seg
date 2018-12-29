#coding:utf-8
from tensorflow.python.framework import ops
import tensorflow as tf
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--height',
                    type = int,
                    default = 704)
parser.add_argument('--width',
                    type = int,
                    default=704)
flags = parser.parse_args()
#####################data prepare#############################
def get_file_names(file_dir):
    imgs=os.listdir(file_dir)
    img_names=[file_dir+'/'+x  for x in imgs]
    label_names=[file_dir+'label/'+x for x in imgs]
    return img_names,label_names

def data_augmentation(image,label,aug=False):
    if aug:
        label = tf.image.resize_images(label, (flags.height, flags.width))
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_hue(image, max_delta=0.05)
        # 设置随机的对比度
        image=tf.image.random_contrast(image,lower=0.99,upper=1.0)

        image_label = tf.concat([image,label],axis = -1)
        maybe_flipped = tf.image.random_flip_left_right(image_label)
        maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)
        #angle=0.4*57
        # angle = np.random.uniform(low=-0.05, high=0.1)
        # maybe_rotate=tf.contrib.image.rotate(maybe_flipped,angle)
        image = maybe_flipped[:, :, :-1]
        label = maybe_flipped[:, :, -1:]
        label=tf.image.resize_images(label, (flags.height/2, flags.width/2))
    else:
        pass
    return image, label
##########################################################
def get_data_label_batch(imgs_dir,labels_dir,augmentation=False):
    imgs_tensor=ops.convert_to_tensor(imgs_dir,dtype=tf.string)

    labels_tensor=ops.convert_to_tensor(labels_dir,dtype=tf.string)
    filename_queue=tf.train.slice_input_producer([imgs_tensor,labels_tensor])

    image_filename = filename_queue[0]
    label_filename = filename_queue[1]

    imgs_values=tf.read_file(image_filename)
    label_values=tf.read_file(label_filename)

    imgs_decorded=tf.image.decode_jpeg(imgs_values,channels=3)
    labels_decorded=tf.image.decode_jpeg(label_values,channels=1)

    imgs_reshaped = tf.image.resize_images(imgs_decorded, (flags.height, flags.width))
    labels_reshaped = tf.image.resize_images(labels_decorded, (flags.height/2, flags.width/2),method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    imgs_reshaped.set_shape([flags.height, flags.width, 3])

    labels_reshaped.set_shape([flags.height/2, flags.width/2, 1])
    imgs_reshaped = tf.cast(imgs_reshaped, tf.float32)
    labels_reshaped = tf.cast(labels_reshaped, tf.float32)
    labels_reshaped = labels_reshaped / (tf.reduce_max(labels_reshaped) + 1e-7)
    # 数据增强
    image, label = data_augmentation(imgs_reshaped, labels_reshaped,aug=augmentation)
    return image, label