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
        loss = -np.sum(np.log(probs[np.arange(0, batch_size), ti.reshape(1, batch_size)]))
        
    else:
        loss = -np.log(probs[target_index])
    
    return loss


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
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")

    #pred = predictions - np.max(predictions)    
    #probs = np.exp(pred) / np.sum(np.exp(pred))
    #loss = -np.log(probs[target_index])
    
    probs = softmax(predictions)
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

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    loss = np.sum(np.sum(W*W)) * reg_strength
    grad = 2*W * reg_strength

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    
    predictions = np.dot(X, W)
    
    probs = softmax(predictions)
    #print("p", probs)
    loss = cross_entropy_loss(probs, target_index)
    
    batch_size = X.shape[0]   
    p = np.zeros(predictions.shape)
    #print(p.shape, X.shape, predictions.shape)
    p[np.arange(0, batch_size), target_index] = 1
    dW = np.dot(X.transpose(), (probs - p))
        
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            #print(len(batches_indices))
            
            for idx in range(len(batches_indices)):                
                # TODO implement generating batches from indices
                batch = X[batches_indices[idx]]
                y_b = y[batches_indices[idx]]                
            
                # Compute loss and gradients
                loss, dW =  linear_softmax(batch, self.W, y_b)
                l2reg, l2grad = l2_regularization(self.W, reg)
            
                # Apply gradient to weights using learning rate
                # Don't forget to add both cross-entropy loss
                # and regularization!
                #print("W", self.W[0:5], "dW", dW[0:5] , "l2gr", l2grad[0:5])
                self.W = self.W - learning_rate * (dW + l2grad)
                # raise Exception("Not implemented!")
                #print(len(batches_indices), idx, loss)

            # end
            loss_history.append(loss)
            #print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        
        res = np.dot(X, self.W)
        #print(res.shape)
        y_pred = np.argmax(res, 1)
        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
