# encoding=utf-8
# CNN-LSTM-CTC-OCR
# Copyright (C) 2017 Jerod Weinman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import tensorflow as tf
import math
import sys
import argparse

from new_invoice_generator import generateImages, generateProperImages

"""Each record within the TFRecord file is a serialized Example proto. 
The Example proto contains the following fields:
  image/encoded: string containing JPEG encoded grayscale image
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/filename: string containing the basename of the image file
  image/labels: list containing the sequence labels for the image text
  image/text: string specifying the human-readable version of the text
"""

# The list (well, string) of valid output characters
# If any example contains a character not found here, an error will result
# from the calls to .index in the decoder below
out_charset="0123456789"

jpeg_data = tf.placeholder(dtype=tf.string)
jpeg_decoder = tf.image.decode_jpeg(jpeg_data,channels=1)

kernel_sizes = [5,5,3,3,3,3] # CNN kernels for image reduction

# Minimum allowable width of image after CNN processing
min_width = 20

def calc_seq_len(image_width):
    """Calculate sequence length of given image after CNN processing"""
    
    conv1_trim =  2 * (kernel_sizes[0] // 2)
    fc6_trim = 2*(kernel_sizes[5] // 2)
    
    after_conv1 = image_width - conv1_trim 
    after_pool1 = after_conv1 // 2
    after_pool2 = after_pool1 // 2
    after_pool4 = after_pool2 - 1 # max without stride
    after_fc6 =  after_pool4 - fc6_trim
    seq_len = 2*after_fc6
    return seq_len

seq_lens = [calc_seq_len(w) for w in range(1024)]

def gen_data(input_base_dir, image_list_filename, output_filebase, 
             num_shards=1000,start_shard=0):
    """ Generate several shards worth of TFRecord data """
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth=True
    sess = tf.Session(config=session_config)
    image_filenames = get_image_filenames(os.path.join(input_base_dir,
                                                       image_list_filename))
    # num_digits = math.ceil( math.log10( num_shards - 1 ))
    num_digits = math.ceil(math.log10(num_shards))
    shard_format = '%0'+ ('%d'%num_digits) + 'd' # Use appropriate # leading zeros
    images_per_shard = int(math.ceil( len(image_filenames) / float(num_shards) ))
    
    for i in range(start_shard,num_shards):
        start = i*images_per_shard
        end   = (i+1)*images_per_shard
        out_filename = output_filebase+'-'+(shard_format % i)+'.tfrecord'
        if os.path.isfile(out_filename): # Don't recreate data if restarting
            continue
        print str(i),'of',str(num_shards),'[',str(start),':',str(end),']',out_filename
        gen_shard(sess, input_base_dir, image_filenames[start:end], out_filename)
    # Clean up writing last shard
    start = num_shards*images_per_shard
    out_filename = output_filebase+'-'+(shard_format % num_shards)+'.tfrecord'
    print str(i),'of',str(num_shards),'[',str(start),':]',out_filename
    gen_shard(sess, input_base_dir, image_filenames[start:], out_filename)

    sess.close()

def gen_shard(sess, input_base_dir, image_filenames, output_filename):
    """Create a TFRecord file from a list of image filenames"""
    writer = tf.python_io.TFRecordWriter(output_filename)
    total = 0
    for filename in image_filenames:
        path_filename = os.path.join(input_base_dir,filename)
        if os.stat(path_filename).st_size == 0:
            print('SKIPPING',filename)
            continue
        try:
            image_data,height,width = get_image(sess,path_filename)
            text,labels = get_text_and_labels(filename)
            if is_writable(width,text):
                example = make_example(filename, image_data, labels, text, 
                                       height, width)
                tmpstr = example.SerializeToString()
                total += len(tmpstr)
                writer.write(tmpstr)
            else:
                print('SKIPPING',filename)
        except:
            # Some files have bogus payloads, catch and note the error, moving on
            print('ERROR',filename)
    print output_filename," total len",total/1024.
    writer.close()


def get_image_filenames(image_list_filename):
    """ Given input file, generate a list of relative filenames"""
    filenames = []
    with open(image_list_filename) as f:
        for line in f:
            # Carve out the ground truth string and file path from lines like:
            # ./2697/6/466_MONIKER_49537.jpg 49537
            filename = line.split(' ',1)[0].split('/')[-1] # split off "./" and number
            filename = filename.strip('\r\n')
            filenames.append(filename)
    return filenames

def get_image(sess,filename):
    """Given path to an image file, load its data and size"""
    with tf.gfile.FastGFile(filename, 'r') as f:
        image_data = f.read()
    image = sess.run(jpeg_decoder,feed_dict={jpeg_data: image_data})
    height = image.shape[0]
    width = image.shape[1]
    return image_data, height, width

def is_writable(image_width,text):
    """Determine whether the CNN-processed image is longer than the string"""
    return (image_width > min_width) and (len(text) <= seq_lens[image_width])
    
def get_text_and_labels(filename):
    """ Extract the human-readable text and label sequence from image filename"""
    # Ground truth string lines embedded within base filename between underscores
    # 2697/6/466_MONIKER_49537.jpg --> MONIKER
    text = os.path.basename(filename).split('_',2)[1][:-4]
    # Transform string text to sequence of indices using charset, e.g.,
    # MONIKER -> [12, 14, 13, 8, 10, 4, 17]
    labels = [out_charset.index(c) for c in list(text)]
    return text,labels

def make_example(filename, image_data, labels, text, height, width):
    """Build an Example proto for an example.
    Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_data: string, JPEG encoding of grayscale image
    labels: integer list, identifiers for the ground truth for the network
    text: string, unique human-readable, e.g. 'dog'
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data)),
        'image/labels': _int64_feature(labels),
        'image/height': _int64_feature([height]),
        'image/width': _int64_feature([width]),
        'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
        'text/string': _bytes_feature(tf.compat.as_bytes(text)),
        'text/length': _int64_feature([len(text)])
    }))
    return example

def _int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("imgtype", help="type of images")
    parser.add_argument("-num_images", help="image number to generate",type=int, default=10000)
    parser.add_argument("-num_shared", help="file to divide",type=int, default=2)
    parser.add_argument("-start_shared", help="start  start", type=int, default=0)

    parser.add_argument("-rfontsize", help="img font type", action="store_true")
    parser.add_argument("-rlength", help="image text length is random", action="store_true")
    parser.add_argument("-rgap", help="image text gap size ", action="store_true")
    parser.add_argument("-rbackgroud", help="image background", action="store_true")
    parser.add_argument("-rotate", help="text rotation", action="store_true")
    parser.add_argument("-rdistortion", help="text distortion", action="store_true")
    parser.add_argument("-rfcolor", help="font front color", action="store_true")
    parser.add_argument("-rfade", help="font  fade", action="store_true")

    args = parser.parse_args()
    return args


def createDirIfNoExist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(argv=None):

    args = parse_arg()
    datatype = args.imgtype
    targetPath='../img/%s'%datatype
    createDirIfNoExist(targetPath)
    namefile='%sanno.txt'%datatype
    tfrPath= '../data/%s/digit'%datatype
    createDirIfNoExist(os.path.dirname(tfrPath))

    num_images = args.num_images
    num_shared = args.num_shared

    # FACTORS = ['RELA_POS', 'FONT', 'FONT_SIZE', 'FONT_GAP', 'FADE', 'ROTATE', 'DISTORTION', 'TEXTLEN', 'BACKGROUND',
    #            'FONTCOLOR']

    factor = ['RELA_POS','FONT','LAYA']
    if args.rfontsize:
        factor.append("FONT_SIZE")
    if args.rlength:
        factor.append("TEXTLEN")
    if args.rgap:
        factor.append("FONT_GAP")
    if args.rbackgroud:
        factor.append("BACKGROUND")
    if args.rotate:
        factor.append("ROTATE")
    if args.rdistortion:
        factor.append("DISTORTION")
    if args.rfcolor:
        factor.append("FONTCOLOR")
    if args.rfade:
        factor.append("FADE")

    # generateImages(datatype,namefile,(256,31),10,num_images)
    generateProperImages(datatype, namefile, (256, 31), 10, num_images, factor)

    gen_data(targetPath, namefile, tfrPath,num_shards=num_shared)
    # gen_data('../data/images', 'annotation_val.txt',   '../data/val/words')
    # gen_data('../data/images', 'annotation_test.txt',  '../data/test/words')

if __name__ == '__main__':
    # main()
    gen_data("/Users/honggh/Desktop/scale2", 'trainiAnaote.txt', '/Users/honggh/github/ocr_project/cnn_lstm_ctc_ocr/data/test/raw', num_shards=1)