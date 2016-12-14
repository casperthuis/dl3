from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import convnet
import time
import tensorflow as tf
import numpy as np
import cifar10_utils
import siamese
import cifar10_siamese_utils
import vgg
from sklearn.manifold import TSNE
from cifar10_siamese_utils import create_dataset
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

# CUSTOM DEFAULTS
SUMMARY_DEFAULT = 0
SAVER_DEFAULT = 0
REG_STRENGTH_DEFAULT = 0.0
DROPOUT_RATE_DEFAULT = 0.0
BATCH_NORM_DEFAULT = 0
FRACTION_SAME_DEFAULT = 0.2
MARGIN_DEFAULT = 0.2



def train_step(loss):
    """
    Defines the ops to conduct an optimization step. You can set a learning
    rate scheduler or pick your favorite optimizer here. This set of operations
    should be applicable to both ConvNet() and Siamese() objects.

    Args:
        loss: scalar float Tensor, full loss = cross_entropy + reg_loss

    Returns:
        train_op: Ops for optimization.
    """
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    optimizer = tf.train.AdamOptimizer

    train_op = optimizer(FLAGS.learning_rate, name='optimizer').minimize(loss) 
    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

def train():
    """
    Performs training and evaluation of ConvNet model.

    First define your graph using class ConvNet and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    Convnn = convnet.ConvNet()
    Convnn.summary = FLAGS.summary
    with tf.name_scope('x'):
        x = tf.placeholder("float", [None, 32,32, 3], name="X_train")
    with tf.name_scope('y'):
        y = tf.placeholder("float", [None, Convnn.n_classes], name="Y_train")

    # initialize graph, accuracy and loss
    logits = Convnn.inference(x)

    loss = Convnn.loss(logits, y)
    accuracy = Convnn.accuracy(logits, y)
    optimizer = train_step(loss)

    init = tf.initialize_all_variables()
    if FLAGS.summary:
        merge = tf.merge_all_summaries()

    if FLAGS.saver:
        saver = tf.train.Saver()


    with tf.Session() as sess:
        sess.run(init)
        cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
        x_test, y_test = cifar10.test.images, cifar10.test.labels

        if FLAGS.summary:
            train_writer = tf.train.SummaryWriter(FLAGS.log_dir + "convnn_train", sess.graph)
            test_writer = tf.train.SummaryWriter(FLAGS.log_dir + "convnn_test")

        for i in range(1, FLAGS.max_steps + 1):
            x_train, y_train = cifar10.train.next_batch(FLAGS.batch_size)

            if FLAGS.summary:
                _, l_train, acc_train, summary = sess.run([optimizer, loss, accuracy, merge],
                                            feed_dict={x: x_train,
                                                       y: y_train,
                                                       Convnn.weight_reg_strength: FLAGS.reg_strength,
                                                       Convnn.dropout_rate: FLAGS.dropout_rate})
                train_writer.add_summary(summary, i)
            else:
                _, l_train, acc_train = sess.run([optimizer, loss, accuracy],
                                            feed_dict={x: x_train, y: y_train,
                                                       Convnn.weight_reg_strength: FLAGS.reg_strength,
                                                       Convnn.dropout_rate: FLAGS.dropout_rate})
 

            if i % FLAGS.eval_freq == 0 or i == 1:
                print("Iteration {0:d}/{1:d}. Train Loss = {2:.3f}, Train Accuracy = {3:.3f}".format(
                    i, FLAGS.max_steps, l_train, acc_train))
                if FLAGS.summary:
                    l_val, acc_val, summary = sess.run([loss, accuracy, merge],
                                          feed_dict={ x: x_test, y: y_test,
                                                      Convnn.weight_reg_strength: FLAGS.reg_strength,
                                                      Convnn.dropout_rate: 0.0})

                    test_writer.add_summary(summary, i)

                else:
                    l_val, acc_val = sess.run([loss, accuracy],
                                          feed_dict={ x: x_test, y: y_test,
                                                      Convnn.weight_reg_strength: FLAGS.reg_strength,
                                                      Convnn.dropout_rate: 0.0})


                print("Iteration {0:d}/{1:d}. Validation Loss = {2:.3f}, Validation Accuracy = {3:.3f}".format(
                    i, FLAGS.max_steps, l_val, acc_val))
        if FLAGS.saver:
            saver.save(sess, FLAGS.checkpoint_dir + '/convnet.ckpt')
    ########################
    # END OF YOUR CODE    #
    ########################


def train_siamese():
    """
    Performs training and evaluation of Siamese model.

    First define your graph using class Siamese and its methods. Then define
    necessary operations such as trainer (train_step in this case), savers
    and summarizers. Finally, initialize your model within a tf.Session and
    do the training.

    ---------------------------
    How to evaluate your model:
    ---------------------------
    On train set, it is fine to monitor loss over minibatches. On the other
    hand, in order to evaluate on test set you will need to create a fixed
    validation set using the data sampling function you implement for siamese
    architecture. What you need to do is to iterate over all minibatches in
    the validation set and calculate the average loss over all minibatches.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every print_freq iterations
    - on test set every eval_freq iterations

    ------------------------
    Additional requirements:
    ------------------------
    Also you are supposed to take snapshots of your model state (i.e. graph,
    weights and etc.) every checkpoint_freq iterations. For this, you should
    study TensorFlow's tf.train.Saver class. For more information, please
    checkout:
    [https://www.tensorflow.org/versions/r0.11/how_tos/variables/index.html]
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)



    # test = cifar10_siamese_utils.create_dataset()

    with tf.name_scope('x'):
        x1 = tf.placeholder("float", [None, 32, 32, 3], name="X_train")
        x2 = tf.placeholder("float", [None, 32, 32, 3], name="X_train")
    with tf.name_scope('y'):
        y = tf.placeholder("float", [None], name="Y_train")

    s = siamese.Siamese()
    channel1 = s.inference(x1)
    channel2 = s.inference(x2, reuse=True)
    loss = s.loss(channel1, channel2, y, FLAGS.margin)
    optimizer = train_step(loss)
    init = tf.initialize_all_variables()

    merge = tf.merge_all_summaries()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        dset_test = cifar10_siamese_utils.create_dataset(source="Test", num_tuples =1000, batch_size = FLAGS.batch_size, fraction_same = FLAGS.fraction_same)
        dset_train = cifar10_siamese_utils.create_dataset(source="Test", num_tuples=FLAGS.max_steps, batch_size=FLAGS.batch_size,
                                                         fraction_same=FLAGS.fraction_same)

        train_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/siamese_train", sess.graph)
        #test_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/siamese_test")

        for i in range(0, FLAGS.max_steps):
            x1_train = dset_train[i][0]
            x2_train = dset_train[i][1]
            y_train = dset_train[i][2]
            feed_dict = {x1: x1_train, x2: x2_train, y: y_train}

            _, l_train, summary = sess.run([optimizer, loss, merge], feed_dict=feed_dict)

            train_writer.add_summary(summary, i)

            if i % EVAL_FREQ_DEFAULT == 0 or i == 1:

                print("Iteration {0:d}/{1:d}. Train Loss = {2:.3f}".format(
                    i, FLAGS.max_steps, l_train))

                test_loss = 0
                for j in range(len(dset_test)):
                    x1_test = dset_test[j][0]
                    x2_test = dset_test[j][1]
                    y_test = dset_test[j][2]
                    feed_dict = {x1: x1_test, x2: x2_test, y: y_test}
                    test_loss += sess.run([loss], feed_dict=feed_dict)[0]

                test_loss = test_loss/len(dset_test)

                #test_writer.add_summary(summary, i)

                print("Iteration {0:d}/{1:d}. Validation Loss = {2:.3f}".format(
                    i, FLAGS.max_steps, test_loss))

        saver.save(sess, FLAGS.checkpoint_dir + '/siamese.ckpt')
    ########################
    # PUT YOUR CODE HERE  #
    ########################



    ########################
    # END OF YOUR CODE    #
    ########################


def feature_extraction():
    """
    This method restores a TensorFlow checkpoint file (.ckpt) and rebuilds inference
    model with restored parameters. From then on you can basically use that model in
    any way you want, for instance, feature extraction, finetuning or as a submodule
    of a larger architecture. However, this method should extract features from a
    specified layer and store them in data files such as '.h5', '.npy'/'.npz'
    depending on your preference. You will use those files later in the assignment.

    Args:
        [optional]
    Returns:
        None
    """

    ########################
    # PUT YOUR CODE HERE  #
    ########################
    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE   #
    ########################
    if FLAGS.train_model == 'linear':
        print("Creating model")
        Convnn = convnet.ConvNet()
        Convnn.summary = FLAGS.summary
        x = tf.placeholder("float", [None, 32, 32, 3], name="X_train")


        # initialize graph, accuracy and loss
        Convnn.inference(x)


        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            print("loading previous session")
            saver.restore(sess, FLAGS.checkpoint_dir + "/convnet.ckpt")
            #saver.restore(sess, FLAGS.checkpoint_dir + "/my_model.cpkt")
            print("Evaluating model")
            cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
            x_test, y_test = cifar10.test.images, cifar10.test.labels

            feed_dict = {x: x_test, Convnn.dropout_rate: 0.0}

            flatten = tf.get_default_graph().get_tensor_by_name("ConvNet/Reshape:0").eval(feed_dict)
            fcl1 = tf.get_default_graph().get_tensor_by_name("ConvNet/fcl1/relu:0").eval(feed_dict)
            fcl2 = tf.get_default_graph().get_tensor_by_name("ConvNet/fcl2/relu:0").eval(feed_dict)

            print("Calculating TSNE")

            _tnse(fcl2, y_test, "conv_fcl2")
            _tnse(fcl1, y_test, "conv_fcl1")
            _tnse(flatten, y_test, "conv_flatten")
    else:

        x1 = tf.placeholder("float", [None, 32, 32, 3], name="X_train")
        s = siamese.Siamese()
        s.inference(x1)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver()
            print("loading previous session")
            saver.restore(sess, FLAGS.checkpoint_dir + "/siamese.ckpt")

            print("Evaluating model")
            cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
            x_test, y_test = cifar10.test.images, cifar10.test.labels

            feed_dict = {x1: x_test}

            l2_norm = tf.get_default_graph().get_tensor_by_name("ConvNet/l2norm:0").eval(feed_dict)

            _tnse(l2_norm, y_test, "siamese_fcl2")

    ########################
    # END OF YOUR CODE    #
    ########################


def _tnse(layer, labels, name):
        
    print("Calculating TSNE for layer%s"%name)
        
    # Create tsne
    #tsne = TSNE(n_components=2, init='pca', random_state=42)
    tsne = TSNE(n_components=2, init='random', random_state=42)
    # Calculate pca
    tsne = tsne.fit_transform(layer)
    # Get predictions
    unnorm = tsne
    # Normalise
    tsne[:,0] = tsne[:,0] + abs(np.min(tsne[:,0]))
    tsne[:,1] = tsne[:,1] + abs(np.min(tsne[:,1]))
    tsne[:,0] = tsne[:,0]/ float(np.max(tsne[:,0]))
    tsne[:,1] = tsne[:,1]/ float(np.max(tsne[:,1]))


    print("Creating figure")
    # create figure
    plt.figure()
    labels = np.argmax(labels, axis=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for i in range(len(classes)):
        class_points = tsne[labels == i]
        plt.scatter(class_points[:,0], class_points[:,1], color=plt.cm.Set1(i*25), alpha=0.5)
    plt.axis([0,1,0,1])
    plt.legend(classes)
    print("Saved image to images/%s.png" %(name))
    plt.savefig('images/%s.png'%name)
    unnorm.dump("tsne_data/%s_tsne.dat"%name)
    labels.dump("tsne_data/%s_labels.dat" % name)
    print("Data dumped in tsne_data/")
    plt.close()


def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

 

def main(_):
    print_flags()

    initialize_folders()
    start = time.time()
    if int(FLAGS.is_train):
        if FLAGS.train_model == 'linear':
            train()
        elif FLAGS.train_model == 'siamese':
            train_siamese()
        else:
            raise ValueError("--train_model argument can be linear or siamese")
    else:
        feature_extraction()
    print("Total run time %i seconds: " %((time.time() - start)))



if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation on the train set')
    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                      help='Frequency of evaluation on the test set')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')
    parser.add_argument('--is_train', type = str, default = True,
                      help='Training or feature extraction')
    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')
    parser.add_argument('--reg_strength', type = float, default = REG_STRENGTH_DEFAULT,
                      help='reg strength')
    parser.add_argument('--dropout_rate', type = float, default = DROPOUT_RATE_DEFAULT,
                      help='dropout rate')
    parser.add_argument('--batch_normal', type = int, default = BATCH_NORM_DEFAULT,
                      help='batch normalisation')
    parser.add_argument('--summary', type=int, default=SUMMARY_DEFAULT,
                        help='summary writer')
    parser.add_argument('--saver', type=int, default=SAVER_DEFAULT,
                        help='save model')
    parser.add_argument('--fraction_same', type=float, default=FRACTION_SAME_DEFAULT,
                        help='fraction same siamese')
    parser.add_argument('--margin', type=float, default=MARGIN_DEFAULT,
                        help='margin siamese loss')


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()


