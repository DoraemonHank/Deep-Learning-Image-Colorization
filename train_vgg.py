# coding=utf-8
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import input_data
import utils as utils
from vgg import vgg16
from model_vgg import ResidualDecoder

MAX_STEPS = 20000
HEIGHT = input_data.HEIGHT
WIDTH = input_data.WIDTH
print(HEIGHT,WIDTH)
BATCH_SIZE = 6
saved_ckpt_path = './checkpoint/'
saved_summary_train_path = './train/'
saved_summary_test_path = './test/'

def train(loss, var_list):
    optimizer = tf.train.AdamOptimizer(0.0001)
    grads = optimizer.compute_gradients(loss, var_list=var_list)
    for grad, var in grads:
        utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


with tf.name_scope("input"):

    x = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, 3], name='x_input')
    y = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH,2], name='ground_truth')
    is_training = tf.placeholder(tf.bool, name="is_training")
    global_step = tf.train.get_or_create_global_step()

# Load vgg16 model
print("🤖 Load vgg16 model...")
vgg = vgg16.Vgg16()

# Build residual encoder model
print("🤖 Build residual encoder model...")
residual_decoder = ResidualDecoder()

with tf.name_scope("vgg16"):
    vgg.build(x)
logits = residual_decoder.build(input_data=x, vgg=vgg, is_training=is_training)


with tf.name_scope("loss"):
# Get loss
    loss = residual_decoder.get_loss(predict_val=logits, real_val=y)
    tf.summary.histogram("loss", loss)

# Prepare optimizer
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name='optimizer')
#optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss)

merged = tf.summary.merge_all()

image_batch, anno_batch, filename = input_data.read_batch(BATCH_SIZE, type = 'train')
image_batch_val, anno_batch_val, filename_val = input_data.read_batch(BATCH_SIZE, type = 'val')

with tf.Session() as sess:

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    #if os.path.exists(saved_ckpt_path):
    ckpt = tf.train.get_checkpoint_state(saved_ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    train_summary_writer = tf.summary.FileWriter(saved_summary_train_path, sess.graph)
    val_summary_writer = tf.summary.FileWriter(saved_summary_test_path, sess.graph)

    for i in range(7, MAX_STEPS + 1):
        print('===================================== epoch ' + str(i) + ' ==============================')
        for j in range((40000//BATCH_SIZE)): 

            b_image, b_anno, b_filename = sess.run([image_batch, anno_batch, filename])
            b_image_val, b_anno_val, b_filename_val = sess.run([image_batch_val, anno_batch_val, filename_val])

            train_summary, _ = sess.run([merged, optimizer], feed_dict={x: b_image, y: b_anno,is_training: True})
            train_summary_writer.add_summary(train_summary, i + j)
            val_summary = sess.run(merged, feed_dict={x: b_image_val, y: b_anno_val,is_training: False})
            val_summary_writer.add_summary(val_summary, i + j)
            
            pred = sess.run(logits, feed_dict={x: b_image_val, y: b_anno_val,is_training: False})

            train_loss_ = sess.run(loss, feed_dict={x: b_image, y: b_anno,is_training: True })
            val_loss = sess.run(loss, feed_dict={x: b_image_val, y: b_anno_val,is_training: False })

            if j % 10 == 0:
                print("training epoch: %d, training loss: %f, val loss: %f" %(i, train_loss_,val_loss))
            if j % 50 == 0:
                print('save image....')
                idx = np.random.randint(0, BATCH_SIZE)
                utils.save_image(b_image_val[idx],pred[idx],'./outputs/result',b_filename_val[idx].decode('utf-8'))
                utils.save_image(b_image_val[idx],b_anno_val[idx],'./outputs/org',b_filename_val[idx].decode('utf-8'))
                saver.save(sess, os.path.join(saved_ckpt_path, str(i) + 'model'), global_step=j)


    coord.request_stop()
    coord.join(threads)