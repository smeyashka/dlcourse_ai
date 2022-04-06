import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    #raise Exception("Not implemented!")
    loss = np.sum(np.sum(W*W)) * reg_strength    
    grad = 2*W * reg_strength
    
    return loss, grad


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


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # TODO copy from the previous assignment
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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        #raise Exception("Not implemented!")
        self.x = X
        mask = X < 0
        res = X.copy()
        res[mask] = 0       
        return res

    
    def backward(self, d_out):
        # TODO copy from the previous assignment
        #raise Exception("Not implemented!")
        dx = self.x.copy()
        mask = self.x < 0
        dx[mask] = 0
        dx[~mask] = 1
        d_result = dx * d_out
        
        #NOTE dx можно сохранить при прямом проходе (он вроде не зависит от параметров)?               
        return d_result

    
    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

        
    def forward(self, X):
        # TODO copy from the previous assignment
        #raise Exception("Not implemented!")
        res = np.dot(X, self.W.value) + np.dot(np.ones((X.shape[0], 1)), self.B.value)
        self.X = X
        return res
    
    
    def backward(self, d_out):
        # TODO copy from the previous assignment
        #raise Exception("Not implemented!")        
        dx = np.dot(d_out, self.W.value.transpose())
        dw = np.dot(self.X.transpose(), d_out)
        db = np.sum(d_out, axis = 0)
        self.W.grad += dw
        self.B.grad += db
        d_input = dx               
        return d_input
    

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
        self.X = None
        
        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = 0
        out_width = 0
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        out_height = height - int(self.filter_size/2)
        out_width = width - int(self.filter_size/2)
        data = np.pad(X, ((0,0), (self.padding, self.padding), 
                          (self.padding, self.padding), (0, 0)), 'constant', constant_values=0)
        self.X = data
        
                
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        #raise Exception("Not implemented!")
        res = np.empty((batch_size, 0)) #np.zeros((batch_size, out_height*out_width*self.out_channels))
        #print(res.shape)
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                #pass
                print(x,y)
                cur = data[:, y : y + self.filter_size, x:x + self.filter_size , : ]
                cur = cur.reshape(batch_size, self.filter_size*self.filter_size*channels)
                w = self.W.value.reshape(self.filter_size*self.filter_size*channels, self.out_channels)
                #cur_b = np.dot(np.ones((cur.shape[0], self.out_channels)), self.B.value)
                cur_b = np.dot(np.ones((cur.shape[0], 1)), self.B.value.reshape(1, self.out_channels))
                print("b", cur_b.shape, cur_b)
                print("c_w", np.dot(cur, w))
                cur_out = np.dot(cur, w) + cur_b
                print("out", cur_out.shape, res, cur_out)
                res = np.append(res, cur_out, axis = 1)
                print(res)
                
        return res.reshape(batch_size, out_height, out_width, self.out_channels)
        


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output        
        
        w = self.W.value.reshape(self.filter_size*self.filter_size*channels, self.out_channels)
        
        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                #pass
                out = d_out[:, y, x, :]
                print("out", out)
                
                dx = np.dot(out, self.W.value.transpose())
                dw = np.dot(self.X.transpose(), out)
                db = np.sum(out, axis = 0)
                print("dx", dx.shape)
                                
                self.W.grad += dw.reshape(self.filter_size, self.filter_size, channels, out_channels)
                self.B.grad += db
                d_input = dx

        #raise Exception("Not implemented!")
        return d_input.reshape(batch_size, height, width, channels)

        
    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        raise Exception("Not implemented!")

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO: Implement backward pass
        raise Exception("Not implemented!")

    def params(self):
        # No params!
        return {}
