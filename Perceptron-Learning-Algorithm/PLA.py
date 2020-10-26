import numpy as np

class BinaryClassifier():
    '''
    Perceptron Learning Algorithm for Binary Classification Problem
    '''
    
    def __init__(self):
        self.train_x = np.array([])
        self.train_y = np.array([])
        self.feature = np.array([])
        self.weights = np.array([])
        self.train_status = 0
        self.test_status = 0
        self.mode = 'train'
    
    def LoadData(self, X = None, Y = None, W = None, mode = 'train'):
        '''
        mode train:
        X is "N data x M Feature"
        Y is "N data and label with {1, -1}"
        W is "M + 1 feature dimension load save Weights, otherwise 'None' is initial to training."
        
        mode predict:
        W is "load save Weights"
        '''
        
        self.mode = mode
        if self.mode == 'train':
            if X is not None:
                self.train_x = np.squeeze(np.array(X)).astype(float)
                self.feature = np.hstack((np.expand_dims(np.ones(len(self.train_x)), axis = 1)
                                         ,self.train_x))
                print('Train_X is loading Complete. Shape is', np.shape(self.train_x),'.')
                print('Feature is loading Complete. Shape is', np.shape(self.feature),'.')
            else:
                raise ValueError("Training Data X is empty.")
            
            if Y is not None:
                self.train_y = np.squeeze(np.array(Y)).astype(int)
                print('Train_Y is loading Complete. Shape is', np.shape(self.train_y),'.')
            else:
                raise ValueError("Label Y is empty.")

            if W is None:
                self.weights = np.zeros(len(self.train_x[0]) + 1).astype(float)
                print('Weights is initial Complete. Shape is', np.shape(self.weights),'.')
            else:
                self.weights = np.squeeze(np.array(W)).astype(float)
                print('Weights is loading Complete. Shape is', np.shape(self.weights),'.')
            
            self.train_status = 1
            print('Training is Ready.')
                
        if self.mode == 'predict':
            if W is not None:
                self.weights = np.squeeze(np.array(W)).astype(float)
                print('Weights is loading Complete. Shape is', np.shape(self.weights),'.')
            else:
                raise ValueError("Weights is empty.")
                    
            self.test_status = 1
            print('Predict is Ready.')
        

    def OutputWeights(self):
        return self.weights
    
    def ScoreFunction(self, X, W):
        return np.sign(np.dot(X, W))
        
    def UpdateWeights(self, X, W, y):
        return np.add(W, np.multiply(y, X))
    
    def NaivePLA(self, iteration = 100):
        if self.train_status:
            for i in range(iteration):
                # Evaluate
                ErrorCount = np.sum(self.ScoreFunction(self.feature, self.weights) != self.train_y)
                print(f'Iteration:{i}')
                print(f'There has {ErrorCount} error points.')

                # Find Error
                ErrorFlag = 0
                for n in range(len(self.feature)):
                    if (self.ScoreFunction(self.feature[n], self.weights) != self.train_y[n]):
                        ErrorFlag = 1
                        self.weights = self.UpdateWeights(self.feature[n], self.weights, self.train_y[n])
                        print(f'Update Weights:{self.weights}')
                        print('------------------------')
                        break
                if ErrorFlag == 0:
                    break
            self.test_status = 1
            return self.weights
        else:
            raise ValueError('Please Load Data First.')
    
    def PocketPLA(self, iteration = 100):
        if self.train_status:
            tmp_weights = np.copy(self.weights)
            FinalErrorCount = np.sum(self.ScoreFunction(self.feature, self.weights) != self.train_y)
            for i in range(iteration):
                # Evaluate
                ErrorCount = np.sum(self.ScoreFunction(self.feature, tmp_weights) != self.train_y)
                print(f'Iteration:{i}')
                print(f'There has {ErrorCount} error points.')
                if FinalErrorCount > ErrorCount:
                    self.weights = np.copy(tmp_weights)
                    FinalErrorCount = ErrorCount
                print(f'Pocket has {FinalErrorCount} error points.')
                
                # Find Error
                ErrorFlag = 0
                for n in range(len(self.feature)):
                    if (self.ScoreFunction(self.feature[n], tmp_weights) != self.train_y[n]):
                        ErrorFlag = 1
                        tmp_weights = self.UpdateWeights(self.feature[n], tmp_weights, self.train_y[n])
                        print(f'Update Temp Weights:{tmp_weights}')
                        print('------------------------')
                        break
                if ErrorFlag == 0:
                    break
            self.test_status = 1
            return self.weights
        else:
            raise ValueError('Please Load Data First.')
            
    def Predict(self, X):
        '''
        Must be 2 Dimension, N Data x M Feature
        '''
        if self.test_status == 1:
            X = np.array(X).astype(float)
            X = np.hstack((np.expand_dims(np.ones(len(X)), axis = 1), X))
            return np.sign(np.dot(X, self.weights))
        else:
            raise ValueError('Please Load Weights or Train first.')