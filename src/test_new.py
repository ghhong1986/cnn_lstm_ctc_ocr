# encoding=utf-8

import os
import time
import tensorflow as tf
from tensorflow.contrib import learn

import mjsynth
import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model','../data/model',
                          """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('output','test',
                          """Sub-directory of model for test summary events""")

tf.app.flags.DEFINE_integer('batch_size',2**8,
                            """Eval batch size""")
tf.app.flags.DEFINE_integer('test_interval_secs', 60,
                             'Time between test runs')

tf.app.flags.DEFINE_string('device','/gpu:0',
                           """Device for graph placement""")

tf.app.flags.DEFINE_string('test_path','../data/',
                           """Base directory for test/validation data""")
tf.app.flags.DEFINE_string('filename_pattern','test/test1-*',
                           """File pattern for input data""")
tf.app.flags.DEFINE_integer('num_input_threads',4,
                          """Number of readers for input data""")

tf.logging.set_verbosity(tf.logging.WARN)

# Non-configurable parameters
mode = learn.ModeKeys.INFER # 'Configure' training mode for dropout layers


def _get_input():
    """Set up and return image, label, width and text tensors"""

    image,width,label,length,text,filename=mjsynth.threaded_input_pipeline(
        FLAGS.test_path,
        str.split(FLAGS.filename_pattern,','),
        batch_size=FLAGS.batch_size,
        num_threads=1,  #FLAGS.num_input_threads,
        num_epochs=None, # Repeat for streaming
        batch_device=FLAGS.device, 
        preprocess_device=FLAGS.device )
    
    return image,width,label,length,filename

def _get_session_config():
    """Setup session config to soften device placement"""
    # per_process_gpu_memory_fraction
    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False,
        gpu_options=gpu_config)

    return config

def _get_testing(rnn_logits,sequence_length):
    """Create ops for testing (all scalars): 
       loss: CTC loss function value, 
       label_error:  Batch-normalized edit distance on beam search max
       sequence_error: Batch-normalized sequence error rate
    """
    # with tf.name_scope("train"):
    #     loss = model.ctc_loss_layer(rnn_logits,label,sequence_length)
    with tf.name_scope("test"):
        predictions,_ = tf.nn.ctc_beam_search_decoder(rnn_logits, 
                                                   sequence_length,
                                                   beam_width=128,
                                                   top_paths=1,
                                                   merge_repeated=False)
        hypothesis = tf.cast(predictions[0], tf.int32) # for edit_distance
        denseHypothesis = tf.sparse_tensor_to_dense(hypothesis,default_value=-1)

    return denseHypothesis

def _get_checkpoint():
    """Get the checkpoint path from the given model output directory"""
    ckpt = tf.train.get_checkpoint_state(FLAGS.model)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path=ckpt.model_checkpoint_path
    else:
        raise RuntimeError('No checkpoint file found')

    return ckpt_path

def _get_init_trained():
    """Return init function to restore trained model from a given checkpoint"""
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_STEP) + 
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    )
    
    init_fn = lambda sess,ckpt_path: saver_reader.restore(sess, ckpt_path)
    return init_fn


outchar = mjsynth.out_charset

def arr2str(arrs):
    carrs = []
    for arr in arrs:
        carr = [ outchar[idx] for idx in arr if idx != -1 ]
        carrs.append(''.join(carr))
    return carrs

def extractValue(name):
    return name.split("_")[1][0:-4]

def main(argv=None):

    with tf.Graph().as_default():
        image,width,label,length,filename = _get_input()

        with tf.device(FLAGS.device):
            features,sequence_length = model.convnet_layers( image, width, mode)
            logits = model.rnn_layers( features, sequence_length,
                                       mjsynth.num_classes() )
            hypothesis = _get_testing(logits,sequence_length)

        global_step = tf.contrib.framework.get_or_create_global_step()

        session_config = _get_session_config()
        restore_model = _get_init_trained()
        
        summary_op = tf.summary.merge_all()
        init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer()) 

        summary_writer = tf.summary.FileWriter( os.path.join(FLAGS.model,
                                                            FLAGS.output) )

        step_ops = [global_step, hypothesis, filename]

        with tf.Session(config=session_config) as sess:
            
            sess.run(init_op)

            coord = tf.train.Coordinator() # Launch reader threads
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            
            summary_writer.add_graph(sess.graph)

            try:            
                while True:
                    restore_model(sess, _get_checkpoint()) # Get latest checkpoint

                    if not coord.should_stop():
                        step,vals,imgfiles = sess.run(step_ops)
                        rightCount = 0
                        count = 0
                        rightset = set()
                        wrongset = set()
                        hyparr = arr2str(vals)
                        for fname,pred in zip(imgfiles,hyparr):
                            actual = extractValue(fname)
                            count += 1
                            if actual == pred :
                                if not actual in rightset:
                                    print 'R' , fname, '--->', pred
                                    rightCount += 1
                                    rightset.add(actual)
                            else:
                                if not actual in wrongset:
                                    print 'W', fname, '--->', pred
                                    wrongset.add(actual)

                        print 'total count %d'%count ,'accury:%f'%(len(rightset)*1./(len(rightset)+len(wrongset)))
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str,step)
                        ## 统计正确率
                    else:
                        break
                    time.sleep(FLAGS.test_interval_secs)
            except tf.errors.OutOfRangeError:
                    print('Done')
            finally:
                coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
