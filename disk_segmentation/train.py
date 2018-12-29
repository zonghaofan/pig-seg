#coding:utf-8

import os
import tensorflow as tf
import numpy as np
import argparse
import pandas as pd
import model
import time
import random
import math

#图片输入尺寸
# h = 1200   #4032
# w = 1600   #3024
h = 768
w = 768
c_image = 3
c_label = 1
image_mean = [127, 129, 128]

parser = argparse.ArgumentParser()
parser.add_argument('--trn_dir',
                    default='./data_trn.csv')

parser.add_argument('--val_dir',
                    default='./data_val.csv')

parser.add_argument('--model_dir',
                    default='./model')

parser.add_argument('--epochs',
                    type=int,
                    default=50)

parser.add_argument('--epochs_per_eval',
                    type=int,
                    default=1)

parser.add_argument('--logdir',
                    default='./logs')

parser.add_argument('--batch_size',
                    type=int,
                    default=16)

parser.add_argument('--learning_rate',
                    type=float,
                    default=1e-3)

parser.add_argument('--decay_rate',
                    type=float,
                    default=0.95)

parser.add_argument('--decay_step',
                    type=int,
                    default=1)

parser.add_argument('--random_seed',
                    type=int,
                    default=1234)

parser.add_argument('--gpu',
                    type=str,
                    default=1)

flags = parser.parse_args()



def set_config():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(flags.gpu)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    # config = tf.ConfigProto(gpu_options=gpu_options)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def blur_image(image, k_size=9):
    kernel = np.ones((k_size, k_size)) / k_size**2
    kernel = tf.reshape(tf.convert_to_tensor(kernel, dtype=tf.float32), [k_size, k_size, 1, 1])

    image = tf.expand_dims(image, axis=0)
    [tR, tG, tB] = tf.unstack(image, num=3, axis=3)
    tR = tf.expand_dims(tR, 3)
    tG = tf.expand_dims(tG, 3)
    tB = tf.expand_dims(tB, 3)

    tR_f = tf.nn.conv2d(tR, kernel, strides=[1, 1, 1, 1], padding='SAME')
    tG_f = tf.nn.conv2d(tG, kernel, strides=[1, 1, 1, 1], padding='SAME')
    tB_f = tf.nn.conv2d(tB, kernel, strides=[1, 1, 1, 1], padding='SAME')

    image_f = tf.stack([tR_f, tG_f, tB_f], axis=3)
    image_f = tf.squeeze(image_f)

    return image_f

def data_augmentation(image, label):
    image_label = tf.concat([image,label], axis=-1)

    maybe_flipped = tf.image.random_flip_left_right(image_label)
    maybe_flipped = tf.image.random_flip_up_down(maybe_flipped)

    image = maybe_flipped[:, :, :-1]
    mask = maybe_flipped[:, :, -1:]
    image = tf.image.random_brightness(image, 0.7)
    # image = tf.image.random_hue(image, 0.3)
    tf.image.random_contrast(image,lower=0.8, upper=1.2)

    return image, mask


def read_csv(queue, augmentation=True):
    csv_reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_content = csv_reader.read(queue)

    image_path, label_path = tf.decode_csv(csv_content, record_defaults=[[""], [""]])

    image_file = tf.read_file(image_path)
    label_file = tf.read_file(label_path)

    try:
        image = tf.image.decode_jpeg(image_file, channels=3)
    except:
        print('Image: ', image_path)
    image = tf.image.resize_images(image, (h, w))
    image.set_shape([h, w, c_image])
    image = tf.cast(image, tf.float32)

    try:
        label = tf.image.decode_jpeg(label_file, channels=1)
    except:
        print('Label: ', label_path)
    label = tf.image.resize_images(label, (h, w))
    label.set_shape([h, w, c_label])

    label = tf.cast(label, tf.float32)
    label = label / (tf.reduce_max(label) + 1e-7)

    image_path = tf.string_split([image_path], '/').values[-1]

    if augmentation:
        image, label = data_augmentation(image, label)
    else:
        pass

    return image, label, image_path


def loss_CE(y_pred, y_true):
    # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=5)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    return cross_entropy_mean


def loss_IOU(y_pred, y_true):
    H, W, _ = y_pred.get_shape().as_list()[1:]
    pred_flat = tf.reshape(y_pred, [-1, H * W])
    true_flat = tf.reshape(y_true, [-1, H * W])
    intersection = tf.reduce_sum(pred_flat * true_flat, axis=1) + 1e-7   #沿着第一维相乘求和
    union = tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(true_flat, axis=1) - intersection + 1e-7
    iou = tf.reduce_mean(intersection / union)
    return 1-iou


def train_op(loss, learning_rate, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.99)
    return optimizer.minimize(loss, global_step=global_step)


def main(flags):
    cfg = set_config()
    current_time = time.strftime("%m/%d/%H/%M/%S")
    trn_logdir = os.path.join(flags.logdir, "trn", current_time)
    val_logdir = os.path.join(flags.logdir, "val", current_time)

    trn = pd.read_csv(flags.trn_dir)
    n_trn = trn.shape[0]
    n_trn_step = n_trn // flags.batch_size

    val = pd.read_csv(flags.val_dir)
    n_val = val.shape[0]
    n_val_step = n_val // flags.batch_size

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, h, w, c_image], name='X')
    y = tf.placeholder(tf.float32, shape=[None, h, w, c_label], name='y')
    mode = tf.placeholder(tf.bool, name='mode')

    # logits = model.unet(X, mode)
    logits = model.depthwise_unet(X, mode)
    pred = tf.nn.sigmoid(logits, name='pred')
    pred_label = tf.cast(tf.greater(pred, 0.5), dtype=tf.float32)

    ce_loss = loss_CE(logits, y)
    iou_loss = loss_IOU(pred, y)
    loss = ce_loss+iou_loss
    tf.summary.scalar("Cross-entropy loss", ce_loss)
    tf.summary.scalar('IOU loss', iou_loss)
    tf.summary.scalar('Total loss', loss)

    H, W = pred_label.get_shape().as_list()[1:3]
    pred_flat = tf.reshape(pred_label, [-1, H * W])
    true_flat = tf.reshape(y, [-1, H * W])

    TP = pred_flat * true_flat
    precision = tf.reduce_sum(TP) / (tf.reduce_sum(pred_flat) + 1e-7)
    recall = tf.reduce_sum(TP) / (tf.reduce_sum(true_flat) + 1e-7)
    f1_score = 2 / (1 / precision + 1 / recall + 1e-7)
    tf.summary.scalar("precision", precision)
    tf.summary.scalar("recall", recall)
    tf.summary.scalar("f1_score", f1_score)

    tf.add_to_collection('inputs', X)
    tf.add_to_collection('inputs', mode)
    tf.add_to_collection('outputs', pred)

    tf.summary.image('Input Image', X, max_outputs=flags.batch_size)
    tf.summary.image('Label', y, max_outputs=flags.batch_size)
    tf.summary.image('Predicted Image', pred, max_outputs=flags.batch_size)

    tf.summary.histogram('Predicted Image', pred)

    # global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
    global_step = tf.train.get_or_create_global_step()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step,
                                               tf.cast(n_trn_step * flags.decay_step, tf.int32),
                                               flags.decay_rate, staircase=True)
    tf.summary.scalar("learning_rate", learning_rate)
    with tf.control_dependencies(update_ops):
        training_op = train_op(loss, learning_rate, global_step)

    summary_op = tf.summary.merge_all()

    # -------------------------------------------- Training -------------------------------------------------

    trn_csv = tf.train.string_input_producer([flags.trn_dir])
    val_csv = tf.train.string_input_producer([flags.val_dir])

    trn_image, trn_label, trn_path = read_csv(trn_csv)
    val_image, val_label, val_path = read_csv(val_csv, augmentation=False)

    X_trn_batch_op, y_trn_batch_op, X_trn_batch_path = tf.train.shuffle_batch([trn_image, trn_label, trn_path],
                                                                              batch_size=flags.batch_size,
                                                                              capacity=flags.batch_size*5,
                                                                              min_after_dequeue=flags.batch_size*2,
                                                                              allow_smaller_final_batch=True)

    X_val_batch_op, y_val_batch_op, X_val_batch_path = tf.train.shuffle_batch([val_image, val_label, val_path],
                                                                              batch_size=flags.batch_size,
                                                                              capacity=flags.batch_size*5,
                                                                              min_after_dequeue=flags.batch_size*2,
                                                                              allow_smaller_final_batch=True)

    with tf.Session(config=cfg) as sess:
        trn_writer = tf.summary.FileWriter(trn_logdir, sess.graph)
        val_writer = tf.summary.FileWriter(val_logdir)

        init = tf.global_variables_initializer()
        sess.run(init)

        # bn_moving_vars = [g for g in tf.global_variables() if 'moving_mean' in g.name or 'moving_variance' in g.name]
        # save_vars = tf.trainable_variables()+bn_moving_vars
        # saver = tf.train.Saver(var_list=save_vars)
        saver = tf.train.Saver()

        model_dir_ = flags.model_dir
        trained_epoch = 0

        # model_dir_ = 'model_v6_16'
        # trained_epoch = int(model_dir_.split('_')[-1])

        if os.path.exists(model_dir_) and tf.train.checkpoint_exists(model_dir_):
            latest_check_point = tf.train.latest_checkpoint(model_dir_)
            saver.restore(sess, latest_check_point)
            print('Load saved model: {}'.format(model_dir_))
            print('Restore training......')
        else:
            print('No model found.')
            print('Start training......')
            try:
                os.rmdir(flags.model_dir)
            except:
                pass
            os.mkdir(flags.model_dir)
        #图文件生成.pb
        #graph = convert_variables_to_constants(sess, , ["pred_bis"])  # pred为保存网络的最后输出节点名称
        # tf.train.write_graph(sess.graph_def, './', 'disk_graph.pb', as_text=False)

        try:
            #global_step = tf.train.get_global_step(sess.graph)

            #使用tf.train.string_input_producer(epoch_size, shuffle=False),会默认将QueueRunner添加到全局图中，
            #我们必须使用tf.train.start_queue_runners(sess=sess)，去启动该线程。要在session当中将该线程开启,不然就会挂起。然后使用coord= tf.train.Coordinator()去做一些线程的同步工作,
            #否则会出现运行到sess.run一直卡住不动的情况。
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for epoch in range(trained_epoch, trained_epoch+flags.epochs):
                for step in range(n_trn_step):
                    print('=================================== training ===================================')
                    # print(sess.run([learning_rate])[0])
                    X_trn, y_trn, X_trn_path = sess.run([X_trn_batch_op, y_trn_batch_op, X_trn_batch_path])
                    # print(X_trn.shape)
                    print(X_trn_path)

                    _, loss_val, step_summary, pre_val, rec_val = sess.run(
                        [training_op, loss, summary_op, precision, recall], feed_dict={X: X_trn, y: y_trn, mode: True})

                    trn_writer.add_summary(step_summary, epoch*n_trn_step + step)
                    print('epoch:{} step:{}/{}, loss: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(epoch+1, step+1, epoch*n_trn_step+step+1, loss_val, pre_val, rec_val))

                for step in range(n_val_step):
                    print('=================================== validation ===================================')
                    X_val, y_val, X_val_path = sess.run([X_val_batch_op, y_val_batch_op, X_val_batch_path])
                    print(X_val_path)
                    loss_val, step_summary, pre_val, rec_val = sess.run([loss, summary_op, precision, recall], feed_dict={X: X_val, y: y_val, mode: False})

                    val_writer.add_summary(step_summary, epoch * n_trn_step + step * n_trn // n_val)
                    print('epoch:{} step:{}, loss: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(epoch + 1, step + 1, loss_val, pre_val, rec_val))

                if (epoch + 1) % 2 == 0:
                    saver.save(sess, '{}/model{:d}.ckpt'.format(flags.model_dir, epoch+1))
                    print('save model {:d}'.format(epoch+1))
                    tf.train.write_graph(sess.graph_def, '', 'disk_graph.pb')

        finally:
            coord.request_stop()
            coord.join(threads)
            saver.save(sess, "{}/model_final.ckpt".format(flags.model_dir))


if __name__ == '__main__':
    main(flags)
