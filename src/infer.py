# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
from tensorflow.contrib import learn
from PIL import Image
import mjsynth
import model
import numpy as np

modelPath = '../data/model'
batch_size =  2**5
device = '/cpu:0'
test_path = '../data/infer'

# FLAGS = tf.app.flags.FLAGS


# Non-configurable parameters
mode = learn.ModeKeys.INFER # 'Configure' training mode for dropout layers



def _get_input():
    """Set up and return image, label, width and text tensors"""

    image, width, label, length, text, filename = mjsynth.threaded_input_pipeline(
        test_path,
        str.split('digit-*', ','),
        batch_size=batch_size,
        num_threads=2,
        num_epochs=None,  # Repeat for streaming
        batch_device=device,
        preprocess_device=device)

    return image, width, label, length


def _get_session_config():
    """Setup session config to soften device placement"""
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)

    return config


def _get_checkpoint():
    ckpt = tf.train.get_checkpoint_state(modelPath)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path = ckpt.model_checkpoint_path
    else:
        raise RuntimeError('No checkpoint file found')

    return ckpt_path


def _get_init_trained():
    """Return init function to restore trained model from a given checkpoint"""
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_STEP) +
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    )

    init_fn = lambda sess, ckpt_path: saver_reader.restore(sess, ckpt_path)
    return init_fn



def get_testing(rnn_logits,sequence_length,label=None):

    # with tf.name_scope("train"):
    #     loss = model.ctc_loss_layer(rnn_logits,label,sequence_length)
    with tf.name_scope("test"):
        predictions,_ = tf.nn.ctc_beam_search_decoder(rnn_logits,
                                                   sequence_length,
                                                   beam_width=128,
                                                   top_paths=1,
                                                   merge_repeated=True)
        hypothesis = tf.cast(predictions[0], tf.int32) # for edit_distance

        denseHypothesis = tf.sparse_tensor_to_dense(hypothesis)

        # label_errors = tf.edit_distance(hypothesis, label, normalize=False)
        # sequence_errors = tf.count_nonzero(label_errors,axis=0)
        # total_label_error = tf.reduce_sum( label_errors )
        # total_labels = tf.reduce_sum( label_length )
        # label_error = tf.truediv( total_label_error,
        #                           tf.cast(total_labels, tf.float32 ),
        #                           name='label_error')
        # sequence_error = tf.truediv( tf.cast( sequence_errors, tf.int32 ),
        #                              tf.shape(label_length)[0],
        #                              name='sequence_error')

    # return loss, label_error, sequence_error
    return denseHypothesis

def get_single_image(path='../data/imgs'):
    imgpathlist = [os.path.join(path, name) for name in os.listdir(path) if name.endswith('jpg')]
    for imgpath in imgpathlist:
        img = Image.open(imgpath).convert("L")
        image =  np.asarray(img)
        shape = list(image.shape)
        width = shape[1]
        shape.append(1)
        image = image.reshape(shape)

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.subtract(image, 0.5)

        # Pad with copy of first row to expand to 32 pixels height
        first_row = tf.slice(image, [0, 0, 0], [1, -1, -1])
        image = tf.concat([first_row, image], 0)

        images = tf.reshape(image,[1,32,width,1])
        widths = tf.convert_to_tensor([width])

        yield images,widths


def main():
    with tf.Graph().as_default():
        # image, width, label, length = _get_input()
        # 直接读取文件内容
        imgs, widths = get_single_image()

        imgsPh = tf.placeholder(tf.float32, [1, 32, None, 1],name='images')  ##不定长度
        widthsPh = tf.placeholder(tf.float32,[1])

        features, sequence_length = model.convnet_layers(imgsPh, widthsPh, mode)
        logits = model.rnn_layers(features, sequence_length,
                                  mjsynth.num_classes())

        hypothesis = get_testing(
            logits, sequence_length)

        session_config = _get_session_config()
        restore_model = _get_init_trained()

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # TODO 最好将输出的结果summary结果中
        step_ops = [hypothesis]

        with tf.Session(config=session_config) as sess:

            sess.run(init_op)
            # restore_model(sess, _get_checkpoint())  # Get latest checkpoint
            # for imgs ,widths,labels in get_single_image():
                # 恢复模型数据
            for i in range(5):
                # hypothesis = sess.run(step_ops,feed_dict={imgsPh:imgs,widthsPh:widths})
                # print hypothesis
                imgs,widths  = sess.run([imgs,widths])
                print widths



def main2():
    with tf.Graph().as_default():

        session_config = _get_session_config()


        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # TODO 最好将输出的结果summary结果中
        # step_ops = [hypothesis]

        with tf.Session(config=session_config) as sess:
            sess.run(init_op)
              # Get latest checkpoint
            for imgs, widths, labels in get_single_image():
                # 恢复模型数据
                # hypothesis = sess.run(step_ops,feed_dict={imgsPh:imgs,widthsPh:widths})
                # print hypothesis

                features, sequence_length = model.convnet_layers(imgs, widths, mode)
                logits = model.rnn_layers(features, sequence_length,
                                          mjsynth.num_classes())

                hypothesis = get_testing(
                    logits, sequence_length)

                restore_model = _get_init_trained()
                restore_model(sess, _get_checkpoint())

                hypo = sess.run([hypothesis])
                print hypo,labels


def testLoadImage():

    # tf.convert_to_tensor()

    with tf.Graph().as_default():
        imgs,widths = get_single_image()

        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)
            iimg  = sess.run(imgs)
            # iimg = iimg.reshape([1, iimg.shape[0], iimg.shape[1], 1])
            # print iimg.shape  ,'w:',width
            print iimg[0,:,0,0]



def test_convert_sparse():
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    indices = [[0, 1], [1, 0]]
    values = [42, 43]
    shape = [2, 2]
    with tf.Session(config=config) as sess:
        sparse_tensor_value = tf.SparseTensorValue(indices, values, shape)
        dense = tf.sparse_tensor_to_dense(sparse_tensor_value)
        dd = sess.run(dense)
        print dd

if __name__ == '__main__':
    # main2()
    main()
    # testLoadImage()
    # test_convert_sparse()