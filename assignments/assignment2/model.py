import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg        
        # TODO Create necessary layers
        #raise Exception("Not implemented!")
        self.lay1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.lay2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        #raise Exception("Not implemented!")
        for val in self.params().values():
            val.grad = np.zeros_like(val.grad)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        res = self.lay1.forward(X)
        res = self.relu.forward(res)
        res = self.lay2.forward(res)
        loss, dpred = softmax_with_cross_entropy(res, y)
        
        dpred = self.lay2.backward(dpred)
        dpred = self.relu.backward(dpred)
        dpred = self.lay1.backward(dpred)      

        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        #raise Exception("Not implemented!")

        for p in self.params().values():
            l2reg, l2grad = l2_regularization(p.value, self.reg)
            p.grad += l2grad
            loss += l2reg
        
        return loss

    
    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        #raise Exception("Not implemented!")
        res = self.lay1.forward(X)
        res = self.relu.forward(res)
        res = self.lay2.forward(res)
        pred = np.argmax(res, 1)
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        #raise Exception("Not implemented!")
        result = {'l1_w' : self.lay1.params()['W'], 'l1_b' : self.lay1.params()['B'],
                  'l2_w' : self.lay2.params()['W'], 'l2_b' : self.lay2.params()['B'],}

        return result
