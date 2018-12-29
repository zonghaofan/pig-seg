#coding:utf-8
from __future__ import print_function
import tensorflow as tf

model_dir = './model_v6_50'
export_path ='./disk_tfserving'

latest_checkpoint = tf.train.latest_checkpoint(model_dir)
saver = tf.train.import_meta_graph(model_dir+'/model_final.ckpt.meta')

X, mode = tf.get_collection('inputs')
print(mode)
print(X)
pred = tf.get_collection('outputs')[0]
print(pred)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
tensor_info_y = tf.saved_model.utils.build_tensor_info(pred)
tensor_info_mode = tf.saved_model.utils.build_tensor_info(mode)

prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'x_input': tensor_info_x,
                'mode_input': tensor_info_mode},
        outputs={'y_output': tensor_info_y},

        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

with tf.Session() as sess:
    # sess.run([tf.local_variables_initializer(), tf.tables_initializer()])
    # sess.run([tf.global_variables_initializer()])

    saver.restore(sess, latest_checkpoint)

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature
        },
    )
    builder.save()

