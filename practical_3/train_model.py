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
from cifar10_siamese_utils import get_cifar10 as get_cifar_10_siamese
from sklearn.manifold import TSNE
from cifar10_siamese_utils import create_dataset
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
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
SUMMARY_DEFAULT = False 
SAVER_DEFAULT = True
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
    Convnn.summary = SUMMARY_DEFAULT
    with tf.name_scope('x'):
        x = tf.placeholder("float", [None, 32,32, 3], name="X_train")
    with tf.name_scope('y'):
        y = tf.placeholder("float", [None, Convnn.n_classes], name="Y_train")

    # initialize graph, accuracy and loss
    logits = Convnn.inference(x)

    loss = Convnn.loss(logits, y)
    accuracy = Convnn.accuracy(logits,y)
    optimizer = train_step(loss)

    init = tf.initialize_all_variables()
    if SUMMARY_DEFAULT:    
        merge = tf.merge_all_summaries()

    if SAVER_DEFAULT:
        saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
        x_test, y_test = cifar10.test.images, cifar10.test.labels
        
        if SUMMARY_DEFAULT:
            train_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/train", sess.graph)
            test_writer = tf.train.SummaryWriter(FLAGS.log_dir + "/test")

        for i in range(1, FLAGS.max_steps + 1):
            x_train, y_train = cifar10.train.next_batch(FLAGS.batch_size)

            _, l_train, acc_train= sess.run([optimizer, loss, accuracy],
                                            feed_dict={x: x_train, y: y_train})
            
            if SUMMARY_DEFAULT:
                _, l_train, acc_train, summary = sess.run([optimizer, loss, accuracy, merge],
                                            feed_dict={x: x_train, y: y_train})
                train_writer.add_summary(summary, i)
            else:
                _, l_train, acc_train = sess.run([optimizer, loss, accuracy],
                                            feed_dict={x: x_train, y: y_train})
 

            if i % EVAL_FREQ_DEFAULT == 0 or i == 1:
                print("Iteration {0:d}/{1:d}. Train Loss = {2:.3f}, Train Accuracy = {3:.3f}".format(
                    i, FLAGS.max_steps, l_train, acc_train))
                if SUMMARY_DEFAULT:
                    l_val, acc_val, summary = sess.run([loss, accuracy, merge], 
                                          feed_dict={ x: x_test, y: y_test})
                
                    test_writer.add_summary(summary, i)

                else:
                    l_val, acc_val = sess.run([loss, accuracy], 
                                          feed_dict={ x: x_test, y: y_test})


                print("Iteration {0:d}/{1:d}. Validation Loss = {2:.3f}, Validation Accuracy = {3:.3f}".format(
                    i, FLAGS.max_steps, l_val, acc_val))
        if SAVER_DEFAULT:

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
    cifar10 = get_cifar_10_siamese('cifar10/cifar-10-batches-py')

    # x, y = cifar10.test.images, cifar10.test.labels
    # val_set = create_dataset([x,y], 100, FLAGS.batch_size, 0.1)


    Siamese = siamese()

    with tf.name_scope('x'):
        x1 = tf.placeholder("float", [None, 32, 32, 3], name="X_train")
        x2 = tf.placeholder("float", [None, 32, 32, 3], name="X_train")
    with tf.name_scope('y'):
        y = tf.placeholder("float", [None, 1], name="Y_train")


    logits = Siamese.inference(x1, )
    loss = Siamese.loss(logits, y)
    optimizer = train_step(loss)
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)
        cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
        x_test, y_test = cifar10.test.images, cifar10.test.labels


        for i in range(1, FLAGS.max_steps + 1):
            x_train, y_train = cifar10.train.next_batch(FLAGS.batch_size)


            _, l_train = sess.run([optimizer, loss],
                                                 feed_dict={x: x_train, y: y_train})

            if i % EVAL_FREQ_DEFAULT == 0 or i == 1:
                print("Iteration {0:d}/{1:d}. Train Loss = {2:.3f}, Train Accuracy = {3:.3f}".format(
                    i, FLAGS.max_steps, l_train, acc_train))
                l_val, acc_val = sess.run([loss, accuracy],
                                              feed_dict={x: x_test, y: y_test})

                print("Iteration {0:d}/{1:d}. Validation Loss = {2:.3f}, Validation Accuracy = {3:.3f}".format(
                    i, FLAGS.max_steps, l_val, acc_val))

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
    # PUT YOUR CODE HERE  #
    ########################
    print("Creating model")
    Convnn = convnet.ConvNet()
    Convnn.summary = SUMMARY_DEFAULT
    with tf.name_scope('x'):
        x = tf.placeholder("float", [None, 32, 32, 3], name="X_train")
    with tf.name_scope('y'):
        y = tf.placeholder("float", [None, Convnn.n_classes], name="Y_train")

    # initialize graph, accuracy and loss
    logits = Convnn.inference(x)
    loss = Convnn.loss(logits, y)
    accuracy = Convnn.accuracy(logits, y)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        print("loading previous session")
        saver.restore(sess, FLAGS.checkpoint_dir + "/convnet.ckpt")
        #saver.restore(sess, FLAGS.checkpoint_dir + "/my_model.cpkt")
        print("Evaluating model")
        cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
        x_test, y_test = cifar10.test.images, cifar10.test.labels
        print(x_test.shape)
        print(y_test.shape)
        l, acc, flatten, fcl1 ,fcl2, logits = sess.run([loss, accuracy,
                                        Convnn.flatten,
                                        Convnn.fcl1,
                                        Convnn.fcl2,
                                        Convnn.logits ],
                                      
                                        feed_dict={x: x_test[0:1000,:,:,:], y: y_test[0:1000]})


        print("Calculating TSNE")
        tnse = TSNE(n_components=2, init='pca', random_state=0)
        pca = tnse.fit_transform(fcl2)
        prediction = np.argmax(logits, axis=1)
        fig = plt.figure()
        
             
      
        classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        #for i in range(Convnn.n_classes):
        #    class_points = pca[prediction == i]
        #    plot = plt.scatter(class_points[:,0], class_points[:,1], color=plt.cm.Set1(i*25), alpha=0.5)
        #    plots.append(plot)
        plt.scatter(pca[:,0], pca[:,1], c=prediction, alpha=0.4)
        plt.legend(tuple(classes))
        plt.savefig('images/tsne_plot.png')
        

        #for label in range(Convnn.n_classes):
        
        #    class_pc = pc[prediction == label]
        #    non_class_pc = pc[prediction != label]
        #    selection_no_class = np.random.choice(len(non_class_pc), len(class_pc))
             

    ########################
    # END OF YOUR CODE    #
    ########################

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
    print("Total run time%i" %((time.time() - start)/60.0))



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

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()


