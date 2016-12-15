from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
import vgg
import convnet
from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers
import cifar10_utils

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
REFINE_AFTER_K_STEPS_DEFAULT = 0

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10/vgg'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

#CUSTOM ARGS
SUMMARY_DEFAULT = 0
SAVER_DEFAULT = 0
REG_STRENGTH_DEFAULT = 0.0
DROPOUT_RATE_DEFAULT = 0.0
BATCH_NORM_DEFAULT = 0
FRACTION_SAME_DEFAULT = 0.2
MARGIN_DEFAULT = 0.2


def train_step(loss, iter):
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
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "FCL/")
    train_op = tf.cond(iter >= FLAGS.refine_after_k,
                       lambda: optimizer(FLAGS.learning_rate, name='optimizer').minimize(loss),
                       lambda: optimizer(FLAGS.learning_rate, name='optimizer').minimize(loss, var_list=train_vars))


    #train_op = optimizer(FLAGS.learning_rate, name='optimizer').minimize(loss, var_list=train_vars)
    # train_op = optimizer(FLAGS.learning_rate, name='optimizer').minimize(loss)

    ########################
    # END OF YOUR CODE    #
    ########################

    return train_op

def train():
    """
    Performs training and evaluation of your model.

    First define your graph using vgg.py with your fully connected layer.
    Then define necessary operations such as trainer (train_step in this case),
    savers and summarizers. Finally, initialize your model within a
    tf.Session and do the training.

    ---------------------------------
    How often to evaluate your model:
    ---------------------------------
    - on training set every PRINT_FREQ iterations
    - on test set every EVAL_FREQ iterations

    ---------------------------
    How to evaluate your model:
    ---------------------------
    Evaluation on test set should be conducted over full batch, i.e. 10k images,
    while it is alright to do it over minibatch for train set.
    """

    # Set the random seeds for reproducibility. DO NOT CHANGE.
    tf.set_random_seed(42)
    np.random.seed(42)


    Convnn = convnet.ConvNet()
    Convnn.summary = True
    ########################
    # PUT YOUR CODE HERE  #
    ########################
    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
    x_test = cifar10.test.images
    with tf.name_scope('x'):        # NONE NONE NONE 3
        x = tf.placeholder("float", [None,None, None, 3], name="X_train")
    with tf.name_scope('y'):
        y = tf.placeholder("float", [None, Convnn.n_classes], name="Y_train")
    with tf.name_scope('iteration'):
        iter = tf.placeholder("float", None, name="iter")


    pool5, assign_ops = vgg.load_pretrained_VGG16_pool5(x, scope_name='vgg')

    with tf.variable_scope('FCL'):
        flatten = tf.reshape(pool5, [-1, pool5.get_shape()[3].value])

        fcl1 = fcl_layer(flatten, [flatten.get_shape()[1].value, 384], 1)

        fcl2 = fcl_layer(fcl1, [fcl1.get_shape()[1].value, 192], 2)

        logits = fcl_layer(fcl2, [fcl2.get_shape()[1].value, 10], 3, last_layer=True)

    loss = Convnn.loss(logits, y)
    accuracy = Convnn.accuracy(logits, y)
    optimizer = train_step(loss, iter)
    init = tf.initialize_all_variables()
    merge = tf.merge_all_summaries()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(assign_ops)

        x_test, y_test = cifar10.test.images, cifar10.test.labels

        train_writer = tf.train.SummaryWriter(FLAGS.log_dir + "vgg_train", sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.log_dir + "vgg_test")

        for i in range(1, FLAGS.max_steps + 1):
            x_train, y_train = cifar10.train.next_batch(FLAGS.batch_size)


            _, l_train, acc_train, summary = sess.run([optimizer, loss, accuracy, merge],
                                                      feed_dict={x: x_train,
                                                                 y: y_train,
                                                                 iter: i})
            train_writer.add_summary(summary, i)


            if i % FLAGS.eval_freq == 0 or i == 1:
                print("Iteration {0:d}/{1:d}. Train Loss = {2:.3f}, Train Accuracy = {3:.3f}".format(
                    i, FLAGS.max_steps, l_train, acc_train))

                l_val, acc_val, summary = sess.run([loss, accuracy, merge],
                                                       feed_dict={x: x_test, y: y_test})

                test_writer.add_summary(summary, i)

                print("Iteration {0:d}/{1:d}. Validation Loss = {2:.3f}, Validation Accuracy = {3:.3f}".format(
                    i, FLAGS.max_steps, l_val, acc_val))

    ########################
    # END OF YOUR CODE    #
    ########################



def fcl_layer(out_p, w_dims, n_layer, last_layer=False):
    """
    Adds a fully connected layer to the graph,
    Args:   out_p: A tensor float containing the output from the previous layer
            w_dims: a vector of ints containing weight dims
            n_layer: an int containing the number of the layer
    """
    with tf.name_scope('fcl%i' % n_layer):
        # Creates weights
        weights = tf.get_variable(
            shape=w_dims,
            initializer=tf.random_normal_initializer(stddev=0.001),
            regularizer=regularizers.l2_regularizer(0.001),
            name="fcl%i/weights" % n_layer)

        # Create bias
        bias = tf.get_variable(
            shape=w_dims[-1],
            initializer=tf.constant_initializer(0.0),
            name="fcl%i/bias" % n_layer)

        # Calculate input

        fcl_out = tf.nn.bias_add(tf.matmul(out_p, weights), bias)

        # Calculate activation
        if not last_layer:
            fcl_out = tf.nn.relu(fcl_out, name="fcl%i" % n_layer)
            fcl_out = tf.nn.dropout(fcl_out, (1.0 - 0.5))

        return fcl_out



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
    train()

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
    parser.add_argument('--refine_after_k', type = int, default = REFINE_AFTER_K_STEPS_DEFAULT,
                      help='Number of steps after which to refine VGG model parameters (default 0).')
    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')
    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')
    parser.add_argument('--reg_strength', type=float, default=REG_STRENGTH_DEFAULT,
                        help='reg strength')
    parser.add_argument('--dropout_rate', type=float, default=DROPOUT_RATE_DEFAULT,
                        help='dropout rate')
    parser.add_argument('--batch_normal', type=int, default=BATCH_NORM_DEFAULT,
                        help='batch normalisation')
    parser.add_argument('--summary', type=int, default=SUMMARY_DEFAULT,
                        help='summary writer')
    parser.add_argument('--saver', type=int, default=SAVER_DEFAULT,
                        help='save model')


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
e