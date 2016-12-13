from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np
import vgg
import convnet

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'
REFINE_AFTER_K_STEPS_DEFAULT = 0

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'

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
    with tf.name_scope('x'):
        x = tf.placeholder("float", [None, 32, 32, 3], name="X_train")
    with tf.name_scope('y'):
        y = tf.placeholder("float", [None, Convnn.n_classes], name="Y_train")

    pool5, _ = vgg.load_pretrained_VGG16_pool5(x, scope_name='vgg')

    flatten = tf.reshape(conv2, [-1, 64 * 8 * 8])
    fcl1 = fcl_layer(flatten, [flatten.get_shape()[1].value, 384], 1)

    fcl2 = fcl_layer(fcl1, [fcl1.get_shape()[1].value, 192], 2)

    logits = fcl_layer(fcl2, [fcl2.get_shape()[1].value, 10], 3, last_layer=True)

    loss = Convnn.loss(logits, y)
    accuracy = Convnn.accuracy(logits, y)
    optimizer = train_step(loss)

    init = tf.initialize_all_variables()
    merge = tf.merge_all_summaries()

    with tf.Session() as sess:
        sess.run(init)

        cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
        x_test, y_test = cifar10.test.images, cifar10.test.labels

        train_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/vgg_train", sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/vgg_test")

        for i in range(1, FLAGS.max_steps + 1):
            x_train, y_train = cifar10.train.next_batch(FLAGS.batch_size)


            _, l_train, acc_train, summary = sess.run([optimizer, loss, accuracy, merge],
                                                      feed_dict={x: x_train,
                                                                 y: y_train})
            train_writer.add_summary(summary, i)

            if i % FLAGS.eval_freq == 0 or i == 1:
                print("Iteration {0:d}/{1:d}. Train Loss = {2:.3f}, Train Accuracy = {3:.3f}".format(
                    i, FLAGS.max_steps, l_train, acc_train))

                l_val, acc_val, summary = sess.run([loss, accuracy, merge],
                                                       feed_dict={x: x_test, y: y_test,
                                                                  Convnn.weight_reg_strength: FLAGS.reg_strength,
                                                                  Convnn.dropout_rate: 0.0})

                test_writer.add_summary(summary, i)


                print("Iteration {0:d}/{1:d}. Validation Loss = {2:.3f}, Validation Accuracy = {3:.3f}".format(
                    i, FLAGS.max_steps, l_val, acc_val))

    ########################
    # END OF YOUR CODE    #
    ########################

    def fcl_layer(self, out_p, w_dims, n_layer, last_layer=False):
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
                initializer=self.fcl_initialiser,
                regularizer=regularizers.l2_regularizer(self.weight_reg_strength),
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
                fcl_out = tf.nn.dropout(fcl_out, (1.0 - self.dropout_rate))
            # Summaries


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


    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
