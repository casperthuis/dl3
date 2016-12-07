from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers

class ConvNet(object):
    """
   This class implements a convolutional neural network in TensorFlow.
   It incorporates a certain graph model to be trained and to be used
   in inference.
    """

    def __init__(self, n_classes = 10):
        """
        Constructor for an ConvNet object. Default values should be used as hints for
        the usage of each parameter.
        Args:
          n_classes: int, number of classes of the classification problem.
                          This number is required in order to specify the
                          output dimensions of the ConvNet.
        """
        self.n_classes = n_classes
        self.weight_reg_strength = 0.000
        self.fcl_initialiser = initializers.xavier_initializer()
        self.conv_initialiser = initializers.xavier_initializer_conv2d()

    def inference(self, x):
        """
        Performs inference given an input tensor. This is the central portion
        of the network where we describe the computation graph. Here an input
        tensor undergoes a series of convolution, pooling and nonlinear operations
        as defined in this method. For the details of the model, please
        see assignment file.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        Although the model(s) which are within the scope of this class do not require
        parameter sharing it is a good practice to use variable scope to encapsulate
        model.

        Args:
          x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]

        Returns:
          logits: 2D float Tensor of size [batch_size, self.n_classes]. Returns
                  the logits outputs (before softmax transformation) of the
                  network. These logits can then be used with loss and accuracy
                  to evaluate the model.
        """
				
        with tf.variable_scope('ConvNet'):
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            #conv1
            conv1 = self._conv_layer(x, [5,5,3,64], 1)
            conv2 = self._conv_layer(conv1, [5,5,64,64], 2)
            flatten_input = tf.reshape(conv2, [-1, 64*8*8])
            fcl1 = self._fcl_layer(flatten_input, [flatten_input.get_shape()[1].value, 384], 1)
            fcl2 = self._fcl_layer(fcl1, [fcl1.get_shape()[1].value, 192], 2)
            logits = self._fcl_layer(fcl2, [fcl2.get_shape()[1].value, 10], 3)
#                weights = tf.get_variable(
#                            name="conv1/weights",
#                            shape= [5 ,5, 3, 64],
#                            initializer=self.conv_initialiser,
#                            regularizer = regularizers.l2_regularizer(self.weight_reg_strength)
#                            )
#
#                bias = tf.get_variable(
#                            name='conv1/bias',
#                            shape= [64],
#                            initializer= tf.constant_initializer(0.0))
#
#                conv = tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding='SAME')
#                relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
#                conv1 = tf.nn.max_pool(relu,
#                                       ksize= [1, 3, 3, 1],
#                                       strides=[1, 2, 2, 1],
#                                       padding='SAME',
#                                       name='pool1')
#       			histogram_summary("conv1", hidden_layer_input)   
            # conv2
#            with tf.name_scope('conv2') as scope:
#                weights = tf.get_variable(
#                    name="conv2/weights",
#                    shape=[5, 5, 64, 64],
#                    initializer=self.conv_initialiser,
#                    regularizer = regularizers.l2_regularizer(self.weight_reg_strength)
#                )
#
#                bias = tf.get_variable(
#                    name='conv2/bias',
#                    shape=[64],
#                    initializer=tf.constant_initializer(0.0))
#
#                conv = tf.nn.conv2d(conv1, weights, [1, 1, 1, 1], padding='SAME')
#                relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
#                conv2 = tf.nn.max_pool(relu,
#                                       ksize=[1, 3, 3, 1],
#                                       strides=[1, 2, 2, 1],
#                                       padding='SAME',
#                                       name='pool')
#
#
#            flatten_input = tf.reshape(conv2, [-1, 64 * 8 * 8])
#
#            with tf.name_scope('fcl1') as scope:
                # print(conv2.get_shape)
                # TODO REMOVE NUMBER HARDCODED
#                flatten_input = tf.reshape(conv2, [-1, 64 * 8 * 8])
#                # print(flatten_input.get_shape())
#                weights = tf.get_variable(
#                    # TODO MIGHT BE THAT THE OUTPUT SIZE COMES FIRST
#                    shape=[flatten_input.get_shape()[1].value, 384],
#                    initializer= self.fcl_initialiser,
#                    regularizer=regularizers.l2_regularizer(self.weight_reg_strength),
#                    name="fcl1/weights")

                # Initialise bias with the settings of FLAGS
#                bias = tf.get_variable(
#                    shape=[384],
#                    initializer=tf.constant_initializer(0.0),
#                    name="fcl1/bias")

#                fcl1_input = tf.nn.bias_add(tf.matmul(flatten_input, weights), bias)
#                fcl1_output = tf.nn.relu(fcl1_input)
#
#            with tf.name_scope('fcl2'):
#
#                weights = tf.get_variable(
#                    # TODO MIGHT BE THAT THE OUTPUT SIZE COMES FIRST
#                    shape=[384, 192],
#                    initializer= self.fcl_initialiser,
#                    regularizer= regularizers.l2_regularizer(self.weight_reg_strength),
#                    name="fcl2/weights")
#
#                # Initialise bias with the settings of FLAGS
#                bias = tf.get_variable(
#                    shape=[192],
#                    initializer=tf.constant_initializer(0.0),
#                    name="fcl2/bias")
#
#                fcl2_input = tf.nn.bias_add(tf.matmul(fcl1_output, weights), bias)
#                fcl2_output = tf.nn.relu(fcl2_input)
#
#                with tf.name_scope('fcl3'):
#                    weights = tf.get_variable(
#                        # TODO MIGHT BE THAT THE OUTPUT SIZE COMES FIRST
#                        shape=[192, 10],
#                        initializer=self.fcl_initialiser,
#                        regularizer=regularizers.l2_regularizer(self.weight_reg_strength),
#                        name="fcl3/weights")

                    # Initialise bias with the settings of FLAGS
#                    bias = tf.get_variable(
#                        shape=[10],
#                        initializer=tf.constant_initializer(0.0),
#                        name="fcl3/bias")
#
#                    logits = tf.nn.bias_add(tf.matmul(fcl2_output, weights), bias)

                ########################
            # END OF YOUR CODE    #
            ########################
        return logits

    def accuracy(self, logits, labels):
        """
        Calculate the prediction accuracy, i.e. the average correct predictions
        of the network.
        As in self.loss above, you can use tf.scalar_summary to save
        scalar summaries of accuracy for later use with the TensorBoard.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                     with one-hot encoding. Ground truth labels for
                     each observation in batch.

        Returns:
          accuracy: scalar float Tensor, the accuracy of predictions,
                    i.e. the average correct predictions over the whole batch.
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

        tf.scalar_summary("accuracy", accuracy) 
        ########################
        # END OF YOUR CODE    #
        ########################
	
        return accuracy

    def loss(self, logits, labels):
        """
        Calculates the multiclass cross-entropy loss from the logits predictions and
        the ground truth labels. The function will also add the regularization
        loss from network weights to the total loss that is return.
        In order to implement this function you should have a look at
        tf.nn.softmax_cross_entropy_with_logits.
        You can use tf.scalar_summary to save scalar summaries of
        cross-entropy loss, regularization loss, and full loss (both summed)
        for use with TensorBoard. This will be useful for compiling your report.

        Args:
          logits: 2D float Tensor of size [batch_size, self.n_classes].
                       The predictions returned through self.inference.
          labels: 2D int Tensor of size [batch_size, self.n_classes]
                       with one-hot encoding. Ground truth labels for each
                       observation in batch.

        Returns:
          loss: scalar float Tensor, full loss = cross_entropy + reg_loss
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

        # The loss
        reg_loss = tf.reduce_sum((tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
				
        full_loss = tf.add(cross_entropy, reg_loss)

        tf.scalar_summary("cross_entropy", cross_entropy)
        tf.scalar_summary("reg_loss", reg_loss)
        tf.scalar_summary("full_loss", full_loss)

        ########################
        # END OF YOUR CODE    #
        ########################

        return full_loss 

	def _conv_layer(self, out_p, w_dims, n_layer):	
        with tf.name_scope('conv%i' %n_layer) as scope:
            # Create weights
            weights = tf.get_variable(name="conv%i/weights" %n_layer,
                                shape= w_dims,
                                initializer= self.conv_initialiser,
                                regularizer = regularizers.l2_regularizer(self.weight_reg_strength))
            
            # Create bias
            bias = tf.get_variable(	name='conv%i/bias' %n_layer,
                                    shape= w_dims[-1],
                                    initializer= tf.constant_initializer(0.0))
            
            # Create input by applying convoltion with the weights on the input
            conv_in = tf.nn.conv2d(out_p, weights, [1, 1, 1, 1], padding='same')
            
            # Add bias and caculate activation
            relu = tf.nn.relu(tf.nn.bias_add(conv_input, bias))
            
            # Apply max pooling
            out = tf.nn.max_pool(	relu,
                                    ksize= [1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='same',
                                    name='pool%i'%n_layer)
            
            # add summary 
            tf.histogram_summary("conv%i/out" %n_layer, conv)
            tf.histogram_summary("conv%i/relu" %n_layer, relu)
            tf.histogram_summary("conv%i/in" %n_layer, conv_in)
            tf.histogram_summary("conv%i/weights"% n_layer, weights)
            tf.histogram_summary("conv%i/bias"% n_layer, bias)
	        
            return out	

    def _fcl_layer(self, out_p, w_dims, n_layer):
        """
        Adds a fully connected layer to the graph,
        Args:   out_p: A tensor float containing the output from the previous layer
                w_dims: a vector of ints containing weight dims
				n_layer: an int containing the number of the layer
        """
        with tf.name_scope('fcl%i'%n_layer) as scope:
            # TODO REMOVE NUMBER HARDCODED
            # Creates weights
            weights = tf.get_variable(
                # TODO MIGHT BE THAT THE OUTPUT SIZE COMES FIRST
                shape=w_dims,
                initializer= self.fcl_initialiser,
                regularizer=regularizers.l2_regularizer(self.weight_reg_strength),
                name="fcl%i/weights"%n_layer)

            # Create bias
            bias = tf.get_variable(
                shape=w_dims[-1],
                initializer=tf.constant_initializer(0.0),
                name="fcl%i/bias"%n_layer)
            
            # Calculate input
            fcl_in = tf.nn.bias_add(tf.matmul(out_p, weights), bias)
            
            # Calculate activation
            fcl_out = tf.nn.relu(fcl_in)

            # Summaries
            tf.histogram_summary("fcl%i/out" %n_layer, fcl_out)
            tf.histogram_summary("fcl%i/in" %n_layer, fcl_in)
            tf.histogram_summary("fcl%i/weights"% n_layer, weights)
            tf.histogram_summary("fcl%i/bias"% n_layer, bias)
            
            return fcl_out
