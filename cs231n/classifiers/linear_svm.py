import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  #num_classes = W.shape[1]
  #num_train = X.shape[0]
  #loss = 0.0
  #for i in xrange(num_train):
  #  di_W = np.zeros(W.shape)
  #  counter = 0
  #  scores = X[i].dot(W)
  #  correct_class_score = scores[y[i]]
  #  for j in xrange(num_classes):
  #    if j == y[i]:
  #      continue
  #    margin = scores[j] - correct_class_score + 1 # note delta = 1
  #    if margin > 0:
  #      loss += margin
  #      di_W[:,j] = X[i]
  #      counter += 1
  #  di_W[:,y[i]] = -counter * X[i]
  #  dW += di_W


  """
  The above code has memory concerns as it uses intermediate variable same dimension as dW.
  Now rewrite using in-place update of dW
  """
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    counter = 0
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        counter += 1
    dW[:,y[i]] += -counter * X[i]
  
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW = dW / num_train + 2 * reg * W # plus the gradient on regularization term
  
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0

  #First, vectorize the inner loop
  #for i in xrange(num_train):
  #  scores = X[i].dot(W)
  #  correct_class_score = scores[y[i]]
  #  margin = scores - correct_class_score + 1
  #  loss += np.sum(np.maximum(margin, 0)) - 1 # "minux one" to subtract "j = y[i]" item from the per-example loss

  # Now, vectorize both loops
  scores = np.dot(X, W)
  margin = np.maximum(0, scores - scores[np.arange(num_train), y].reshape(num_train,1) + 1)
  margin[np.arange(num_train), y] = 0 # correction of margin of true class label
  loss = np.sum(margin)
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  multiplier = np.zeros(margin.shape)
  multiplier[margin > 0] = 1
  multiplier[np.arange(num_train), y] = -np.sum(multiplier, axis=1)
  dW = np.dot(X.T, multiplier)
  dW = dW / num_train + 2 * reg * W # plus the gradient on regularization term
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
