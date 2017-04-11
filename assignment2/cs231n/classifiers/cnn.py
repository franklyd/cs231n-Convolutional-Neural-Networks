import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - batch-norm - relu - 2x2 max pool - affine - batch-norm - relu - dropout - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    (C, H, W) = input_dim 
    self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
    self.params['W2'] = np.random.randn(num_filters * (H/2) * (W/2), hidden_dim) * weight_scale
    self.params['W3'] = np.random.randn(hidden_dim,num_classes) * weight_scale

    self.params['b1'] = np.zeros(num_filters)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['b3'] = np.zeros(num_classes)

    self.params['gamma1'] = np.ones(C)
    self.params['beta1'] = np.ones(C)
    self.params['gamma2'] = np.ones(hidden_dim)
    self.params['beta2'] = np.ones(hidden_dim)

    self.bn_params = [{'mode': 'train'} for i in range(2)]
    self.dropout_param = {'mode': 'train', 'p': 0.5}

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    mode = 'test' if y is None else 'train'
    for bn_param in self.bn_params:
        bn_param[mode] = mode
    self.dropout_param['mode'] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    ## forward pass: 
    # 1. conv_relu_pool_forward, and put the result in a single column
    pool_out, pool_cache = self.conv_bn_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, self.bn_params[0], pool_param)
    pool_shape = pool_out.shape
    pool_out = pool_out.reshape(X.shape[0],W2.shape[0])  # don't forget to unfold it
    # 2. affine_relu_forward
    aff1_out, aff1_cache = self.affine_bn_relu_forward(pool_out, W2, b2, gamma2, beta2, self.bn_params[0])
    # 3. dropout
    aff1_out, drop_cache = dropout_forward(aff1_out, self.dropout_param)
    # 4. affine
    aff2_out, aff2_cache = affine_forward(aff1_out, W3, b3)
    # 5. final scores 
    scores = aff2_out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    ## If training mode, compute softmax loss
    loss, dout = softmax_loss(scores, y)
    # add L2 regulation
    loss += 0.5 * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)) * self.reg
    
    ## Backpropagate: calculate grad
    # 4. affine
    dout, grads['W3'], grads['b3'] = affine_backward(dout, aff2_cache)
    # 3. dropout
    dout = dropout_backward(dout, drop_cache)
    # 2. affine_relu
    dout, grads['W2'], grads['b2'] = self.affine_bn_relu_backward(dout, aff1_cache)
    # 1. conv_relu__pool
    # fold dout
    dout = dout.reshape(pool_shape)
    _, grads['W1'], grads['b1'] = self.conv_bn_relu_pool_backward(dout, pool_cache)

    ## regulation gradient
    #for i in range(3):
     #   grads['W'+str(i+1)] += self.reg * self.params['W'+str(i+1)]
    grads['W1'] += self.reg * W1
    grads['W2'] += self.reg * W2
    grads['W3'] += self.reg * W3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  def affine_bn_relu_forward(self,x, w, b,gamma,beta,bn_param):

    a, fc_cache = affine_forward(x, w, b)
    bn_a, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn_a)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


  def affine_bn_relu_backward(self,dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    dbn_a = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(dbn_a, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta

  def conv_bn_relu_pool_forward(self, x, w, b, gamma, beta, conv_param, bn_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    n, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    s, relu_cache = relu_forward(n)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache 


  def conv_bn_relu_pool_backward(self, dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    dn, dgamma, dbeta = spatial_batchnorm_backward(ds, bn_cache)
    da = relu_backward(dn, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta

pass
