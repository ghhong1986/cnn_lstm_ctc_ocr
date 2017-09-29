# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import mjsynth
def test_bucket():
    seq_lengths = np.array([6, 3, 2])
    inputs = []
    inputs.append(tf.convert_to_tensor(np.array([2,3,3,3,3,3])))
    inputs.append(tf.convert_to_tensor(np.array([2, 3, 4])))
    inputs.append(tf.convert_to_tensor(np.array([12, 13, 14])))
    inputs.append(tf.convert_to_tensor(np.array([3, 4])))
    inputs.append(tf.convert_to_tensor(np.array([13, 4])))

    sequences, output = tf.contrib.training.bucket_by_sequence_length(input_length=seq_lengths, tensors= inputs, batch_size=2, bucket_boundaries =[1, 2], allow_smaller_final_batch=True,
                                                  dynamic_pad=True, capacity=2)

    init_op = tf.initialize_all_variables()

    sess = tf.Session()

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            s, o= sess.run([sequences, output])
            print s
            print o
            print '$%%^^&*('*3

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def test_train_bucket():
    train_path = '../data/train'
    # image, width, label, _, _, _ = mjsynth.bucketed_input_pipeline(
    #     train_path,
    #     ['digit-0*'],
    #     batch_size=8,
    #     num_threads=1,
    #     input_device='/cpu:0',
    #     width_threshold=50,
    #     length_threshold=10)


    # data_queue = mjsynth._get_data_queue(train_path,["dig*"],
    #                              capacity=10)
    # image, width, label, length, text, filename = mjsynth._read_word_record(
    #     data_queue)

    image, width, label, length, text, _ = mjsynth.bucketed_input_pipeline(
        train_path,
        ["dig*"],
        batch_size=4,
        num_threads=2,
        input_device="/cpu:0",
        width_threshold=None,
        length_threshold=None)

    init_op = tf.global_variables_initializer()

    sess = tf.Session()

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            s, o,w,t = sess.run([image, label,width,text])
            print s.shape
            print s[:,:,s.shape[2]-1,0]
            # print o, '\n======\n'
            print w,'\n',t,'\n',o

            print '$%%^^&*(' * 3
            # break
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

# test_train_bucket()

def test_tf_batch_read():
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

        # tf.train.batch(label)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)
        # Create a coordinator and run all QueueRunner objects
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for batch_index in range(20):
            img, lbl = sess.run([image, text])
            print lbl, img.shape
            img = img.reshape((img.shape[0:2]))
            pimg = Image.fromarray(img, "L")


        # Stop the threads
        coord.request_stop()

        # Wait for threads to stop
        coord.join(threads)

        sess.close()

test_bucket()