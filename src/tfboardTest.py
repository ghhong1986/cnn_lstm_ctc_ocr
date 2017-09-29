# -*- coding: utf-8 -*-


import tensorflow as tf
from PIL import Image
import numpy as np
imgpath = '../fortest/1.jpg'
img = Image.open(imgpath).convert("L")
imgarr = np.asarray(img)


weight = np.random.rand(3,1,3,2)


loss = np.random.randint(100, size=10)


def graph():
    with tf.Graph().as_default():
        timg = tf.image.convert_image_dtype(imgarr, tf.uint8)
        tweight = tf.convert_to_tensor(weight)

        nweight = tf.squeeze(tweight,axis=1)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # global_step = tf.contrib.framework.get_or_create_global_step()
        # summary_writer = tf.summary.FileWriter("../tmp")
        ops = [timg,tweight]


        #

        with tf.Session() as sess:
            sess.run(init_op)
            # summary_writer.add_graph(sess.graph)

            # 创建一个graph

            nwe  =sess.run(nweight)
            print nwe.shape
            print nwe

sess = tf.Session()

def test_fun(sess):
    loss = [234,120,249,334]
    widths = tf.convert_to_tensor(loss,dtype=tf.int32)
    conv1_trim = tf.constant(2, dtype=tf.int32, name="conv1_trim")
    one = tf.constant(1, dtype=tf.int32, name="one")
    two = tf.constant(2, dtype=tf.int32, name="two")
    after_sub = tf.subtract(widths, conv1_trim)
    after_div = tf.floor_div(after_sub, two)
    after_sub2 = tf.subtract(after_div, one)
    #
    shape_b = tf.shape(after_sub2)
    sequence_length = tf.reshape(after_sub2, [-1], name='seq_len')
    shape_a = tf.shape(sequence_length)
    seq_len,width,b,a = sess.run([sequence_length,widths,shape_b,shape_a])
    print b,a
    return seq_len,width

seq ,width =  test_fun(sess)
print width
print seq

