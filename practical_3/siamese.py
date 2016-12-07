from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



class Siamese(object):
    """
    This class implements a siamese convolutional neural network in
    TensorFlow. Term siamese is used to refer to architectures which
    incorporate two branches of convolutional networks parametrized
    identically (i.e. weights are shared). These graphs accept two
    input tensors and a label in general.
    """

    def inference(self, x, reuse = False):
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
        with tf.variable_scope('ConvNet') as conv_scope:
            ########################
            # PUT YOUR CODE HERE  #
            ########################
            conv1 = self._conv_layer(x, [5, 5, 3, 64], 1, reuse=reuse)
            conv2 = self._conv_layer(conv1, [5, 5, 64, 64], 2, reuse=reuse)
            flatten_input = tf.reshape(conv2, [-1, 64 * 8 * 8], reuse=reuse)
            fcl1 = self._fcl_layer(flatten_input, [flatten_input.get_shape()[1].value, 384], 1, reuse=reuse)
            l2_out = self._fcl_layer(fcl1, [fcl1.get_shape()[1].value, 192], 2, last_layer=True, reuse=reuse)

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
        raise NotImplementedError
        ########################
        # END OF YOUR CODE    #
        ########################

        return loss


    def _conv_layer(self, out_p, w_dims, n_layer, reuse):
        with tf.name_scope('conv%i' % n_layer):
            # Create weights
            weights = tf.get_variable(name="conv%i/weights" % n_layer,
                                      shape=w_dims,
                                      initializer=self.conv_initialiser)

            # Create bias
            bias = tf.get_variable(name='conv%i/bias' % n_layer,
                                   shape=w_dims[-1],
                                   initializer=tf.constant_initializer(0.0))

            # Create input by applying convoltion with the weights on the input
            conv_in = tf.nn.conv2d(out_p, weights, [1, 1, 1, 1], padding='SAME')

            # Add bias and caculate activation
            relu = tf.nn.relu(tf.nn.bias_add(conv_in, bias))

            # Apply max pooling
            out = tf.nn.max_pool(relu,
                                 ksize=[1, 3, 3, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 name='pool%i' % n_layer)

            # add summary
            if self.summary:
                pass
                # tf.histogram_summary("conv%i/out" %n_layer, out)
                # tf.histogram_summary("conv%i/relu" %n_layer, relu)
                # tf.histogram_summary("conv%i/in" %n_layer, conv_in)
                # tf.histogram_summary("conv%i/weights"% n_layer, weights)
                # tf.histogram_summary("conv%i/bias"% n_layer, bias)

            return out

    def _fcl_layer(self, out_p, w_dims, n_layer, last_layer=False, reuse=False):

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
                name="fcl%i/weights" % n_layer)


            # Create bias
            bias = tf.get_variable(
                shape=w_dims[-1],
                initializer=tf.constant_initializer(0.0),
                name="fcl%i/bias" % n_layer)

            # Calculate input

            fcl_out = tf.nn.bias_add(tf.matmul(out_p, weights), bias)
            fcl_out = tf.nn.relu(fcl_out, name="fcl%i" % n_layer)
            # Calculate activation
            if last_layer:
               fcl = tf.nn.l2_normalize(fcl_out, dim=0)

            # Summaries
            if self.summary:
                pass
                # tf.histogram_summary("fcl%i/out" %n_layer, fcl_out)
                # tf.histogram_summary("fcl%i/weights"% n_layer, weights)
                # tf.histogram_summary("fcl%i/bias"% n_layer, bias)

            return fcl_out

