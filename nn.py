import pandas as pd
import numpy as np
class Layer:
    def __init__(self, in_size, out_size, activation, keep_prob=1.0, params_initialization='random'):
        
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        self.keep_prob = keep_prob
        self.params_init = params_initialization
        self.input = None
        self.dw = None
        self.db = None
        self.out = None
        self.z = None
        self.__init_params__()
        
    def __init_params__(self):
        if self.params_init == "random":
            self.w = np.random.uniform(size=(self.out_size, self.in_size)) * 0.1 - 0.05
        elif self.params_init == 'lecun':
            self.w = np.random.uniform(low=-1/np.sqrt(self.in_size), high=1/np.sqrt(self.in_size), size=(self.out_size, self.in_size))
        elif self.params_init == 'xe':
            self.w = np.random.uniform(low=-np.sqrt(2/(self.in_size+self.out_size)), high=np.sqrt(2/(self.in_size+self.out_size)), size=(self.out_size, self.in_size))
        elif self.params_init == 'he':
            self.w = np.random.uniform(low=-np.sqrt(2/(self.in_size)), high=np.sqrt(2/(self.in_size)), size=(self.out_size, self.in_size))
        self.b = np.zeros((self.out_size, 1))
        
    def __call__(self, X, train=False):
        
        assert self.in_size == X.shape[0]
        
        self.input = X
        
        self.z = np.matmul(self.w, X) + self.b
        
        if train:
            dropout = np.random.uniform(low=0, high=1, size=(self.out_size, 1)) < self.keep_prob
            self.z = np.multiply(self.z, dropout)
        
        self.out = self.activation(self.z)
        
        return self.out
    
    def __back__(self, a_next, m, final_layer, lr):
        '''
        a_prev = np.matmul(w_prev.T, dz_prev)
        '''
        if final_layer:
            dz = a_next
        else:
            dz = a_next * self.activation(self.z, back_prop=True)
        
        self.dw = np.matmul(dz, self.input.T) / m
        self.db = np.sum(dz, axis=1, keepdims=True) / m
        
        # self.w = self.w - lr * self.dw
        # self.b = self.b - lr * self.db
        
        return np.matmul(self.w.T, dz)
    
    def __update_params__(self, lr):
        
        # print((self.w - lr * self.dw == self.w).sum())
        # print(np.max(self.dw))
        # print(np.max(self.db))
        self.w = self.w - lr * self.dw
        self.b = self.b - lr * self.db
        
class Model:
    def __init__(self, loss_func):
        self.layers = []
        self.num_layer = 0
        self.loss_func = loss_func
        self.loss_log = []
        self.accuracy_log = []
        self.val_loss_log = []
        self.val_accuracy_log = []
    
    def add_layer(self, layer):
        if self.num_layer > 0:
            assert self.layers[-1].out_size == layer.in_size
        
        self.layers.append(layer)
        self.num_layer += 1
        
    def __call__(self, X, train=False):
        
        out = X
        
        for layer in self.layers:
            out = layer(out, train=train)
        
        return out
    
    def __back__(self, y_true, y_pred, lr):
        
        assert y_true.shape == y_pred.shape
        
        a_prev = y_pred - y_true
        final_layer = True
        
        m = y_true.shape[1]
        
        for layer in reversed(self.layers):
            a_prev = layer.__back__(a_prev, m, final_layer=final_layer, lr=lr)
            final_layer = False
        
        for layer in self.layers:
            layer.__update_params__(lr)
    
    def fit(self, X, y, iter=30, lr=0.1, val_set=None):
        
        for i in range(iter):
    
            out = self.__call__(X, train=True)
            self.__back__(y, out, lr)
            
            loss = self.loss_func(y, out)
            acc = self.evaluate(X, y)
            
            self.loss_log.append(loss)
            self.accuracy_log.append(acc)
            
            if val_set is not None:
                val_X = val_set[0]
                val_y = val_set[1]
                
                val_out = self.__call__(val_X)
                val_loss = self.loss_func(val_y, val_out)
                val_accuracy = self.evaluate(val_X, val_y)
                
                self.val_loss_log.append(val_loss)
                self.val_accuracy_log.append(val_accuracy)

                print(f"epoch: {i+1}/{iter}\tloss: {loss}\taccuracy:{acc}\tval_loss: {val_loss}\tval_accuracy: {val_accuracy}")
            else:
                print(f"epoch: {i+1}/{iter}\t\tloss: {loss}\t\taccuracy:{acc}")
            
    def evaluate(self, X, y, return_accuracy=False):
        
        m = X.shape[1]
        out = self.__call__(X)
        
        y_true = np.argmax(y, axis=0)
        y_pred = np.argmax(out, axis=0)
        
        return (y_true == y_pred).sum() / m
            
def ReLU(X, back_prop=False):
    
    if back_prop:
        return np.where(X>0, 1, 0)
    
    return np.maximum(0, X)


def softmax(X, back_prop=False):
    '''
    X: (number of units, number of instances)
    
    return (units, number of instances)
    '''
    
    # max_val = np.max(X, axis=1, keepdims=True)
    
    # X = X - max_val
    
    X = np.clip(X, -500, 500)
    
    divisor = np.exp(X).sum(axis=0)
    
    return np.exp(X) / divisor

def sigmoid(X, back_prop=False):
    
    if back_prop:
        sig = sigmoid(X)
        return sig * (1 - sig)
    
    return 1/(1 + np.exp(-X))

def tanh(X, back_prop=False):
    
    if back_prop:
        th = tanh(X)
        return 1 - np.power(th, 2)
    
    # np.clip(X, -250, 250)
    
    return np.tanh(X)

def binary_cross_entropy(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    epsilon=1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) 
    
    loss = - y_true * np.log(y_pred) - (1-y_true) * np.log(1-y_pred)
    
    return loss.mean()

def categorical_cross_entropy(y_true, y_pred):
    '''
    y_true: (number of output classes, total number of instances)
    '''
    
    assert y_true.shape == y_pred.shape
    epsilon=1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) 
    loss = - (y_true * np.log(y_pred)).sum(axis=0)
    # loss = (- (y_true) * np.log(y_pred + epsilon)).sum(axis=0)
    
    return loss.mean()

def accuracy(y_true, y_pred):
    
    assert y_true.shape == y_pred.shape
    
    m = y_true.shape[1]
    
    y_true = y_true.argmax(axis=0)
    y_pred = y_pred.argmax(axis=0)
    
    return (y_true == y_pred).sum() / m