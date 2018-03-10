'''
SGD算法 产考：http://blog.csdn.net/pengjian444/article/details/71075544#reply
numpy
'''
import numpy as np
import matplotlib.pyplot as plt
import array

def gen_line_data(sample_num = 100):
    '''

    :param sample_num:产生的样本大小为100
    :return:
    '''
    x_1 = np.linspace(0, 9, sample_num)  #生成线性空间 x_1
    ''' 
    plt.plot(x_1) #画出x_1图像
    plt.show()
    '''
    x_2 = np.linspace(4, 13, sample_num)
    #将[x_1] [x_2]连接
    x = np.concatenate([x_1], [x_2], axis=0).T  #the shape x is : (100,2)
    y = np.dot(x, np.array([3, 4]).T) # y列向量 ;.dot为矩阵乘法 shape y is :(100,)
    return x, y

def sgd(samples, y, step_size=0.01, max_iter_count=10000):
    '''
    随机梯度下降法
    :param samples:采样样本
    :param y: 结果->
    :param step_size:迭代步长
    :param max_iter_count:最大迭代数； batch_size：块大小
    :return:
    '''
    samples_num, dim = samples.shape
    '''
    a = [[1, 3], [2, 4], [3, 4]]
    a = array(a)    #numpy array()
    a.flatten() #res=[1,3,2,4,3,4]
    '''
    y = y.flatten()
    w = np.ones((dim,), dtype=np.float32)
    loss = 10
    iter_count = 0
    while loss > 0.001 and iter_count < max_iter_count:
        loss = 0
        error = np.zeros((dim,), dtype=np.float32)
        for i in range(samples_num):
            predict_y = np.dot(w.T, samples[i])
            for j in range(dim):
                error[j] += (y[i] - predict_y)*samples[i][j]
                w[j] += step_size * error[j] / samples_num


        for i in range(samples_num):
            predict_y = np.dot(w.T, samples[i])
            error = (1 / (samples_num * dim)) * np.power((predict_y - y[i]), 2)
            loss += error

        print("ite_count: ", iter_count, "the loss:", loss)
        iter_count += 1
    return w


if __name__ == '__main__':
    samples, y = gen_line_data()
    w = sgd(samples, y)
    print(w)



