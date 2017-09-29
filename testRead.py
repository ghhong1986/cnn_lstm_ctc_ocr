# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

data_path = '../data/t1/digit-1.tfrecord'

# data_path = '../data/train/words-000.tfrecord'

# data_path = 'train.tfrecords'  # address to save the hdf5 file

# http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/

feature_map = {
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                        default_value=''),
    'image/labels': tf.VarLenFeature(dtype=tf.int64),
    'image/width': tf.FixedLenFeature([1], dtype=tf.int64,
                                      default_value=1),
    'image/filename': tf.FixedLenFeature([], dtype=tf.string,
                                         default_value=''),
    'text/string': tf.FixedLenFeature([], dtype=tf.string,
                                      default_value=''),
    'text/length': tf.FixedLenFeature([1], dtype=tf.int64,
                                      default_value=1)
}

def batchReadTf(tfpath):

    with tf.Session() as sess:
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([tfpath], num_epochs=1)
        # Define a reader and read the next record
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        # Decode the record read by the reader
        features = tf.parse_single_example(serialized_example, features=feature_map)

        image = tf.image.decode_jpeg(features['image/encoded'], channels=1)  # gray
        width = tf.cast(features['image/width'], tf.int32)  # for ctc_loss
        label = tf.serialize_sparse(features['image/labels'])  # for batching
        length = features['text/length']
        text = features['text/string']
        filename = features['image/filename']

        # print text

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for batch_index in range(20):
            img, lbl = sess.run([image, text])
            print lbl,img.shape
            img = img.reshape((img.shape[0:2]))
            pimg = Image.fromarray(img, "L")

            # pimg.save("../img/%d-%s.jpg" % (batch_index,lbl))
            # img = img.astype(np.uint8)
            # for j in range(6):
            #     plt.subplot(2, 3, j + 1)
            #     plt.imshow(img[j, ...])        example.features.feature['image/encoded']

            #     plt.title('cat' if lbl[j] == 0 else 'dog')
            # plt.show()
        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)

        sess.close()


#
# example = tf.train.Example(features=tf.train.Features(feature={
#     'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data)),
#     'image/labels': _int64_feature(labels),
#     'image/height': _int64_feature([height]),
#     'image/width': _int64_feature([width]),
#     'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
#     'text/string': _bytes_feature(tf.compat.as_bytes(text)),
#     'text/length': _int64_feature([len(text)])
# }))

def tfrecordIter(tfName):
    record_iterator = tf.python_io.tf_record_iterator(path=tfName)
    idx =0
    for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)

        img_str = (example.features.feature['image/encoded']
                                  .bytes_list
                                  .value[0])


        label  = example.features.feature['image/labels'].int64_list

        height = int(example.features.feature['image/height']
                     .int64_list
                     .value[0])

        width = int(example.features.feature['image/width']
                    .int64_list
                    .value[0])

        text = example.features.feature["text/string"].bytes_list.value[0]


        # img_1d = np.fromstring(img_str, dtype=np.uint8)
        # reconstructed_img = img_1d.reshape((height, width, -1))
        #
        # pimg = Image.fromarray(reconstructed_img, "L")
        #
        # pimg.save("img/%d-%s.jpg" % (idx,label))

        # img_string = (example.features.feature['image_raw']
        #               .bytes_list
        #               .value[0])
        #
        # annotation_string = (example.features.feature['mask_raw']
        #                      .bytes_list
        #                      .value[0])

        print text,width,height
        # break

# tfrecordIter(data_path)

batchReadTf(data_path)