from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import initializers
from tensorflow.contrib.layers import regularizers
from tensorflow.contrib.layers import flatten


class Siamese(object):
    """
    This class implements a siamese convolutional neural network in
    TensorFlow. Term siamese is used to refer to architectures which
    incorporate two branches of convolutional networks parametrized
    identically (i.e. weights are shared). These graphs accept two
    input tensors and a label in general.
    """

    def inference(self, x, reuse=False):
        """
        Defines the model used for inference. Output of this model is fed to the
        objective (or loss) function defined for the task.

        Here we recommend you to consider using variable and name scopes in order
        to make your graph more intelligible for later references in TensorBoard
        and so on. You can define a name scope for the whole model or for each
        operator group (e.g. conv+pool+relu) individually to group them by name.
        Variable scopes are essential components in TensorFlow for parameter sharing.
        You can use the variable scope to activate/deactivate 'variable reuse'.

        Args:
           x: 4D float Tensor of size [batch_size, input_height, input_width, input_channels]
           reuse: Python bool to switch reusing on/off.

        Returns:
           l2_out: L2-normalized output tensor of shape [batch_size, 192]

        Hint: Parameter reuse indicates whether the inference graph should use
        parameter sharing or not. You can study how to implement parameter sharing
        in TensorFlow from the following sources:

        https://www.tensorflow.org/versions/r0.11/how_tos/variable_scope/index.html
        """
        with tf.variable_scope('ConvNet', reuse=reuse) as conv_scope:
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            conv1 = self._conv_layer(x, [5, 5, 3, 64], 1)
            conv2 = self._conv_layer(conv1, [5, 5, 64, 64], 2)
            flatten_input = tf.reshape(conv2, [-1, 64 * 8 * 8])
            fcl1 = self._fcl_layer(flatten_input, [flatten_input.get_shape()[1].value, 384], 1)
            fcl2 = self._fcl_layer(fcl1, [fcl1.get_shape()[1].value, 192], 2)
            with tf.name_scope("l2norm") as scope:
                l2_out = tf.nn.l2_normalize(fcl2, 0, name=scope)
            ########################
            # END OF YOUR CODE    #
            ########################

        return l2_out

    def loss(self, channel_1, channel_2, label, margin):
        """
        Defines the contrastive loss. This loss ties the outputs of
        the branches to compute the following:

               L =  Y * d^2 + (1-Y) * max(margin - d^2, 0)

               where d is the L2 distance between the given
               input pair s.t. d = ||x_1 - x_2||_2 and Y is
               label associated with the pair of input tensors.
               Y is 1 if the inputs belong to the same class in
               CIFAR10 and is 0 otherwise.

               For more information please see:
               http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf

        Args:
            channel_1: output of first channel (i.e. branch_1),
                              tensor of size [batch_size, 192]
            channel_2: output of second channel (i.e. branch_2),
                              tensor of size [batch_size, 192]
            label: Tensor of shape [batch_size]
            margin: Margin of the contrastive loss

        Returns:
            loss: scalar float Tensor
        """
        ########################
        # PUT YOUR CODE HERE  #
        ########################
        d = tf.reduce_sum(tf.square(channel_1 - channel_2), 1)
        #d_sqrt = tf.sqrt(d)
        right_part = tf.mul((1 - label), tf.maximum(0., margin - d))
        left_part = tf.mul(label, d)
        contrastive_loss = tf.reduce_mean(tf.add(right_part, left_part))
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        full_loss = contrastive_loss + reg_loss
        tf.scalar_summary("cross_entropy", contrastive_loss)
        tf.scalar_summary("reg_loss", reg_loss)
        tf.scalar_summary("full_loss", full_loss)
        ########################
        # END OF YOUR CODE    #
        ########################

        return full_loss


    def _conv_layer(self, out_p, w_dims, n_layer):
        conv_stride = [1, 1, 1, 1]
        maxp_shape = [1, 3, 3, 1]
        maxp_stride = [1, 2, 2, 1]

        weight_init = tf.random_normal_initializer(stddev=0.001)
        bias_init = tf.constant_initializer(0.0)

        # TUNING MODEL:
        weight_reg = regularizers.l2_regularizer(0.0)


        with tf.name_scope('Conv%i' % n_layer) as scope:
            # Create weights

            weights = tf.get_variable(name='Conv%i/' % n_layer+"weigths",
                                      shape=w_dims,
                                      initializer=weight_init)

            # Create bias
            bias = tf.get_variable(name='Conv%i/' % n_layer+"bias",
                                   shape=w_dims[-1],
                                   initializer=bias_init)

            # Create input by applying convoltion with the weights on the input
            conv_in = tf.nn.conv2d(out_p, weights,
                                   conv_stride,
                                   padding='SAME',
                                   name= scope+"conv")

            # Add bias and caculate activation
            relu = tf.nn.relu(tf.nn.bias_add(conv_in, bias),
                              name=scope+"relu")

            # Apply max pooling
            out = tf.nn.max_pool(relu,
                                 ksize=maxp_shape,
                                 strides=maxp_stride,
                                 padding='SAME',
                                 name=scope+"pool")

        return out

    def _fcl_layer(self, out_p, w_dims, n_layer):

        """
        Adds a fully connected layer to the graph,
        Args:   out_p: A tensor float containing the output from the previous layer
                w_dims: a vector of ints containing weight dims

                n_layer: an int containing the number of the layer

        return:
                fcl_out: the output of the fully connected layer
        """

        weight_init = tf.random_normal_initializer(stddev=0.001)
        bias_init = tf.constant_initializer(0.)

        # TUNING MODEL:
        weight_reg = regularizers.l2_regularizer(0.000)
        dropout_rate = 0.0

        with tf.name_scope('fcl%i' % n_layer) as scope:
            # Creates weights
            weights = tf.get_variable(
                shape=w_dims,
                initializer=weight_init,
                regularizer=weight_reg,
                name= 'fcl%i/' % n_layer+"weights")

            # Create bias
            bias = tf.get_variable(
                shape=w_dims[-1],
                initializer=bias_init,
                name= 'fcl%i/'% n_layer+ "bias")

            # Calculate input
            fcl_in = tf.nn.bias_add(tf.matmul(out_p, weights), bias, name=scope+"input")
            fcl_out = tf.nn.relu(fcl_in, name=scope+"output")

            # Calculate activation
            fcl_out = tf.nn.dropout(fcl_out, 1 - dropout_rate)


        return fcl_out

