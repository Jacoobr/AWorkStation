# coding:utf-8

'''
@author: xiaojianli
@sotfware: PyCharm Community Edition
@explanation: 
@time
steps:
1. calculate distance
2. find the minimum distance
'''
import numpy as np
class NearestNeibor:
    def _init__(self):
        pass
    #train the model
    def train_model(self, X, Y):
        '''
        :param X: N x D
        :param Y: 1-dimention N
        :return:
        '''
        self.Xtr = X
        self.ytr = Y

    def predict(self, X):
        '''
        :param X: 2-dimentions N x D
        :return:
        '''
        num_test = X.shape[0]  #N
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        #get the minimum distance for each element to training set element (using Manhattan distance), get the [i].ytr
        for i in range(num_test):
            '''
            X[i, :] is the every rows of 2-dimention array X
            '''
            distance = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            print("X[i, :]", X[i, :])
            min_distance = np.argmin(distance)  #get the index of nearest distance element
            print("min_distance: ", min_distance)
            Ypred[i] = self.ytr[min_distance]   #get the label
        return Ypred

if __name__ == '__main__':
    nn = NearestNeibor()
    train_x = np.random.rand(4, 2)
    train_y = np.random.randint(0, 9, 4)    # sample 4 number: 0~9
    pre_x = np.random.rand(4, 2)
    print("train_y: ", train_x)
    print("train_y: ", train_y)
    print("pre_x: ", pre_x)
    nn.train_model(train_x, train_y)
    print(nn.predict(pre_x))




