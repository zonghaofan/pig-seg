#coding:utf-8
from __future__ import print_function
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_dir = './Model_pig_BN'
export_path ='./datase'

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./Model_pig_BN/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./Model_pig_BN'))
    X, mode = tf.get_collection('inputs')
    print(mode)
    print(X)
    pred = tf.get_collection('pred')[0]
    print(pred)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(pred)
    tensor_info_mode=tf.saved_model.utils.build_tensor_info(mode)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x_input': tensor_info_x,
                    'mode_input': tensor_info_mode},
            outputs={'y_output': tensor_info_y},

            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature
        },
    )
    builder.save()

