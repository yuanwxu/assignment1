import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = np.dot(X[i], W)
    correct_label_score = scores[y[i]]
    prob = 1.0 / np.sum(np.exp(scores - correct_label_score))
    loss += - np.log(prob)
    for j in xrange(num_classes):
      if j == y[i]:
        dW[:,j] += -X[i] * (1 - prob)
      else:
        dW[:,j] += X[i] * (1.0 / np.sum(np.exp(scores - scores[j])))
      
  loss = loss / num_train + reg * np.sum(W * W)
  dW = dW / num_train + 2* reg * W
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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = np.dot(X, W)
  correct_label_score = scores[np.arange(num_train), y]
  prob_correct = 1.0 / np.sum(np.exp(scores - correct_label_score.reshape(num_train,1)), axis=1)
  loss = - np.sum(np.log(prob_correct))
  loss = loss / num_train + reg * np.sum(W * W)

  # To vectorize gradient, think X transpose (dim D-by-N) times a "multiple" of dim N-by-C, so
  # get right dim of W. Work out what this "multiple" looks like by writing a few terms explicitly
  
  scores = scores - np.amax(scores, axis=1, keepdims=True) # subtract maximum of each row for numerical stability
  scores_exp = np.exp(scores)

  # Each row of "multiple" (dim N-by-C) is a vector with elem -(1-prob_correct_i) at position y_i and prob_j at position j (i is index for example and j index for label)
  multiple = scores_exp / np.sum(scores_exp, axis=1, keepdims=True) 
  #multiple[np.arange(num_train), y] = - (1 - multiple[np.arange(num_train), y]) # this is also correct
  
  multiple[np.arange(num_train), y] = - (1 - prob_correct)
  dW = np.dot(X.T, multiple)
  dW = dW / num_train + 2 * reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

