#coding:utf-8
import cv2
import numpy as np
import model
import vgg16
import tensorflow as tf
import os
import pandas as pd
import argparse
from data_prepare import get_file_names,get_data_label_batch

#label的size是输入图片的一半大小
#图片输入尺寸
img_size = 704
label_size = img_size / 2
c_image = 3
c_label = 1

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir',
                    default = './data/train')
parser.add_argument('--test_dir',
                    default = './data/test')
parser.add_argument('--epochs',
                    type = int,
                    default = 15)
parser.add_argument('--batch_size',
                    type = int,
                    default = 8)
parser.add_argument('--model_dir',
                    default = './model')
parser.add_argument('--learning_rate',
                    type = float,
                    default = 1e-6)
#衰减系数
parser.add_argument('--decay_rate',
                    type = float,
                    default = 0.9)
#衰减速度
parser.add_argument('--decay_step',
                    type = int,
                    default = 100)

flags = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def data_augmentation(image,label,aug=False):
    if aug:
        label = tf.image.resize_images(label, (img_size, img_size))
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
        label=tf.image.resize_images(label, (label_size, label_size))

    else:
        pass
    return image, label

def read_csv(queue,augmentation=False):

    csv_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_content = csv_reader.read(queue)

    image_path, label_path = tf.decode_csv(csv_content,record_defaults=[[""],[""]])

    image_file = tf.read_file(image_path)

    label_file = tf.read_file(label_path)

    image = tf.image.decode_jpeg(image_file, channels = 3)
    image=tf.image.resize_images(image,(img_size,img_size))
    image.set_shape([img_size,img_size,c_image])
    image = tf.cast(image, tf.float32)

    label = tf.image.decode_jpeg(label_file, channels = 1)
    label = tf.image.resize_images(label, (label_size, label_size))
    label.set_shape([label_size,label_size,c_label])

    label = tf.cast(label,tf.float32)
    label = label / (tf.reduce_max(label) + 1e-7)
    #数据增强
    image,label = data_augmentation(image,label,augmentation)

    return image,label,image_path,label_path
def del_certain_num(x):
    y = [i for i in range(1, 11)]
    if x in y:
        y.remove(x)
    return y
def data_loader(i):
    # for i in range(1,11):
        # print(i)
    flags.test_dir = './data/train' + str(i)
    test_imgs_dir, test_labels_dir = get_file_names(flags.test_dir)
    num_test = len(test_imgs_dir)
    print('num_test=', num_test)
    train_imgs_dir = []
    train_labels_dir = []
    for j in del_certain_num(i):
        flags.train_dir = './data/train' + str(j)
        train_imgs_dir_, train_labels_dir_ = get_file_names(flags.train_dir)
        train_imgs_dir.extend(train_imgs_dir_)
        train_labels_dir.extend(train_labels_dir_)
    num_train = len(train_imgs_dir)
    print('num_train=', num_train)
    train_img, train_label = get_data_label_batch(train_imgs_dir, train_labels_dir, augmentation=True)
    # print(train_img)

    test_img, test_label = get_data_label_batch(test_imgs_dir, test_labels_dir, augmentation=False)
    # print(train_img)

    # batch_size是返回的一个batch样本集的样本个数。capacity是队列中的容量
    X_train_batch_op, y_train_batch_op = tf.train.shuffle_batch([train_img, train_label],
                                                                batch_size=flags.batch_size,
                                                                capacity=flags.batch_size * 5,
                                                                min_after_dequeue=flags.batch_size * 2,
                                                                allow_smaller_final_batch=True)

    print(X_train_batch_op, y_train_batch_op)
    X_test_batch_op, y_test_batch_op = tf.train.shuffle_batch([test_img, test_label],
                                                              batch_size=flags.batch_size,
                                                              capacity=flags.batch_size * 5,
                                                              min_after_dequeue=flags.batch_size * 2,
                                                              allow_smaller_final_batch=True)

    print(X_test_batch_op)
    return num_train,num_test,X_train_batch_op,y_train_batch_op,X_test_batch_op,y_test_batch_op
def main(flags):
    tf.reset_default_graph()

    model1 = model.Model()
    model1.build_model()
    pred = model1.Prob_C
    inputs = model1.input_holder
    # outputs = model1.Prob

    tf.add_to_collection('inputs', inputs)
    tf.add_to_collection('inputs', model1.is_training)
    tf.add_to_collection('pred', pred)

    tf.summary.scalar('loss total',model1.Loss_total)
    tf.summary.scalar('loss iou',model1.C_IoU_LOSS)
    tf.summary.scalar('precision', model1.precision)
    tf.summary.scalar('recall', model1.recall)
    tf.summary.scalar('f1-score', model1.f1_score)

    tf.summary.image('image',model1.input_holder,max_outputs=flags.batch_size)

    a = tf.reshape(model1.label_holder,(-1,label_size, label_size, 2))
    tf.summary.image('label', tf.expand_dims(a[:,:,:,0],axis=-1),max_outputs=flags.batch_size)
    tf.summary.image('pred',tf.expand_dims(pred[:,:,:,0],axis=-1),max_outputs=flags.batch_size)

    global_steps = tf.train.get_or_create_global_step()
    # learning_rate = tf.train.exponential_decay(flags.learning_rate, global_steps,
    #                                            num_train//flags.batch_size,
    #                                            flags.decay_rate, staircase=True)

    cycle = tf.cast(tf.floor(1. + tf.cast(global_steps, dtype=tf.float32) / (2 * 2000.)), dtype=tf.float32)

    x = tf.cast(tf.abs(tf.cast(global_steps, dtype=tf.float32) / 2000. - 2. * cycle + 1.), dtype=tf.float32)

    learning_rate = 1e-6 + (1e-3 - 1e-6) * tf.maximum(0., (1 - x)) / tf.cast(2 ** (cycle - 1), dtype=tf.float32)
    tf.summary.scalar('learning_rate', learning_rate)
    summary_op = tf.summary.merge_all()

    tvars = tf.trainable_variables()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    opt = tf.train.AdamOptimizer(learning_rate)
    with tf.control_dependencies(update_ops):
        # 设置最大梯度防止梯度爆炸
        max_grad_norm = 1
        # 梯度裁剪防止梯度爆炸
        grads, _ = tf.clip_by_global_norm(tf.gradients(model1.Loss_total, tvars), max_grad_norm)
        train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_steps)
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('./logs/test')

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=flags.epochs)
        if os.path.exists('Model_pig_BN') and tf.train.checkpoint_exists('Model_pig_BN'):
            latest_check_point = tf.train.latest_checkpoint('Model_pig_BN')
            saver.restore(sess, latest_check_point)
        # if os.path.exists('Model_pig_BN'):
        #     """恢复模型权重文件"""
        #     """notice"""
        #     saver.restore(sess, './Model_pig_BN/model66.ckpt')
        else:
            print('No model')
        i=1
        try:
            for epoch in range(flags.epochs):
                num_train, num_test, X_train_batch_op, y_train_batch_op, X_test_batch_op, y_test_batch_op = data_loader(i)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                if i < 10:
                    i+=1
                else:
                    i=1
                for step in range(0, num_train, flags.batch_size):
                    X_train, y_train = sess.run([X_train_batch_op, y_train_batch_op])
                    print('===================train=============================')
                    print('X_train.shape',X_train.shape)

                    y_train = np.stack((y_train, 1 - y_train), axis=3)

                    y_train = np.reshape(y_train, [-1, label_size*label_size,2])
                    print('y_train',y_train.shape)
                    _, loss, loss_iou, step_summary, acc,_,_,global_steps_value= sess.run(
                        [train_op, model1.Loss_total, model1.C_IoU_LOSS, summary_op,
                         model1.precision,model1.recall,model1.f1_score,global_steps],
                                                    feed_dict={model1.input_holder: X_train,
                                                               model1.label_holder:y_train,
                                                               model1.is_training: True})
                    print("Epoch={},step={} loss_mean={},Accuracy={}".format(epoch,global_steps_value, loss, acc))
                    train_writer.add_summary(step_summary, global_steps_value)
                for step in range(0, num_test, flags.batch_size):
                    print('===================test=============================')
                    X_test, y_test = sess.run([X_test_batch_op, y_test_batch_op])
                    y_test = np.stack((y_test, 1 - y_test), axis=3)
                    y_test = np.reshape(y_test, [-1, label_size * label_size, 2])
                    _, loss, loss_iou, step_summary, acc,_,_ = sess.run(
                        [train_op, model1.Loss_total, model1.C_IoU_LOSS, summary_op,
                         model1.precision,model1.recall,model1.f1_score,],
                        feed_dict={model1.input_holder: X_test,
                                   model1.label_holder: y_test,
                                   model1.is_training: False})
                    """notice"""
                    test_step=epoch * (num_train // flags.batch_size) + step // flags.batch_size * num_train // num_test
                    print("Epoch={},step={} loss_mean={},Accuracy={}".format(epoch,test_step, loss, acc))
                    test_writer.add_summary(step_summary, test_step)
                if epoch % 1 == 0:
                    saver.save(sess, "Model_pig_BN/model{}.ckpt".format(epoch))
                    print('save model{}'.format(epoch))
                    tf.train.write_graph(sess.graph_def, '', 'pig_graph.pb')
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, './Model_pig_BN/model.ckpt')


if __name__ == "__main__":
    main(flags)
