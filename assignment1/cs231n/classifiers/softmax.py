import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  num_dim = X.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    scores_origin = X[i].dot(W)
    scores = scores_origin - np.max(scores_origin)
    correct_class_score = scores[y[i]]
    soft_max_loss = np.exp(correct_class_score) / np.sum(np.exp(scores))
    X_i = X[i].reshape((num_dim,1))
    dW += X_i.dot(np.exp(scores_origin.reshape(1,num_classes))) / np.sum(np.exp(scores_origin)) 
    dW[:,y[i]] -= X[i]
    loss += -np.log(soft_max_loss) 
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores_origin = X.dot(W)
  scores = scores_origin - np.max(scores_origin,axis=1).reshape((num_train,1))
  correct_class_score = scores[np.arange(num_train),y]
  soft_max_loss = np.divide(np.exp(correct_class_score) , np.sum(np.exp(scores),axis=1))
  loss = np.sum(-np.log(soft_max_loss))
  dW1 = np.divide(np.exp(scores_origin),np.sum(np.exp(scores_origin),axis=1).reshape((num_train,1)))
  dW1[np.arange(num_train),y] += -1
  dW = X.T.dot(dW1)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W
  return loss, dW

