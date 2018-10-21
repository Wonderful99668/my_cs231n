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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  for i in range(num_train):
    scores = np.dot(X[i], W)
    scores -= np.max(scores)
    loss += -scores[y[i]] + np.log(np.sum(np.exp(scores)))
    for j in range(num_class):
      soft_max = np.exp(scores[j]) / np.sum(np.exp(scores))
      if j == y[i]:
        dW[:, j] += (-1 + soft_max) * X[i]
      else:
        dW[:, j] += soft_max * X[i]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W ** 2)
  dW = dW / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = np.dot(X, W)
  shifted_scores = scores - np.max(scores, axis=1).reshape(num_train, 1)
  # loss = -np.sum(shifted_scores[np.arange(num_train), y]) + np.sum(np.log(np.sum(np.exp(shifted_scores), axis=1)))
  soft_max_mid = np.exp(shifted_scores) / (np.sum(np.exp(shifted_scores), axis=1).reshape(num_train, 1))
  soft_max = soft_max_mid[np.arange(num_train), y]
  loss = -np.sum(np.log(soft_max))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W ** 2)

  dW_mid = soft_max_mid.copy()
  dW_mid[np.arange(num_train), y] += -1
  dW = np.dot(X.T, dW_mid)
  dW = dW / num_train + reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

