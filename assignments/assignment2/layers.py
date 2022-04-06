import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    
    if predictions.ndim != 1:
        batch_size = predictions.shape[0]
        N = predictions.shape[1]
        pred = predictions - np.max(predictions, 1).reshape(1, batch_size).transpose()    
        #sigma = np.divide(np.exp(pred), np.sum(np.exp(pred), 1)).reshape(1, predictions.shape[0]).transpose() 
        sigma = np.divide(np.exp(pred), np.sum(np.exp(pred), 1).reshape(batch_size,1).dot(np.ones((1,N))))
    else:
        pred = predictions - np.max(predictions)
        sigma = np.exp(pred) / np.sum(np.exp(pred)) 
    #print("sigma", sigma) 
    return sigma



def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
       
    if probs.ndim != 1:
        #loss = -np.log(probs[:, target_index])
        batch_size = probs.shape[0]        
        ti = target_index.copy()
        q = probs[np.arange(0, batch_size), ti.reshape(1, batch_size)]
        #q[q < 1e-20] = 1e-20       
        loss = -np.sum(np.log(q))        
        #print("ti", ti, "probs", probs[np.arange(0, batch_size), ti.reshape(1, batch_size)])
    else:
        loss = -np.log(probs[target_index])
    
    return loss


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    #raise Exception("Not implemented!")    
    loss = np.sum(np.sum(W*W)) * reg_strength    
    grad = 2*W * reg_strength
    
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    #raise Exception("Not implemented!")
    probs = softmax(preds)
    #print("p", probs)
    loss = cross_entropy_loss(probs, target_index)
    #print("l", loss)
    
    dprediction = probs
    
    if dprediction.ndim != 1:        
        batch_size = probs.shape[0]
        ti = target_index.copy()
        dprediction[np.arange(0, batch_size), ti.reshape(1, batch_size)] -= 1        
    else:
        dprediction[target_index] -= 1
    

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):       
        self.x = None        
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        #raise Exception("Not implemented!")
        self.x = X
        mask = X < 0
        res = X.copy()
        res[mask] = 0       
        return res
        

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        dx = self.x.copy()
        mask = self.x < 0
        dx[mask] = 0
        dx[~mask] = 1
        d_result = dx * d_out
        
        #NOTE dx можно сохранить при прямом проходе (он вроде не зависит от параметров)?
        
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        res = np.dot(X, self.W.value) + np.dot(np.ones((X.shape[0], 1)), self.B.value)
        self.X = X
        return res

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        #raise Exception("Not implemented!")
        dx = np.dot(d_out, self.W.value.transpose())
        dw = np.dot(self.X.transpose(), d_out)
        db = np.sum(d_out, axis = 0)
        self.W.grad += dw
        self.B.grad += db
        d_input = dx
        
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
