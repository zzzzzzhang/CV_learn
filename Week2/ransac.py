
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

#model:linear fit

def genSamples():
    '''
    to generate some samples :0~100 and 50% with -100~100 noise
    '''
    x = np.arange(100)
    y = np.arange(100) + np.random.random(100) * np.random.randint(-1,2,100)
    ranomchose = np.random.randint(100,size= 50)
    for i in ranomchose:
        y[i] += np.random.randint(-100, 101)
        
    return x, y

def inner(y, yfit, err = 2):
    '''
    calculate number of inner points
    reurn: num of inner points,index of inner points 
    '''
    d = abs(y - yfit)
    idx = np.where(d < err)[0]
    return len(idx), idx

def ransac(x, y, num= 10, epoch = 100):
    '''
    choose num points randomly for each epoch
    return: model
    '''
    n_inner = 0
    model = None
    
    for i in range(epoch):
        randomChoose = np.random.randint(0,len(x),num)
        randomChoose.sort()
        x_choosed = x[randomChoose]
        y_choosed = y[randomChoose]
        p = np.polyfit(x_choosed, y_choosed, 1)
        f = np.poly1d(p)
        yfit = f(x)
        n, idx = inner(y, yfit, err = 2)
        if n > n_inner:
            print('num of inner points is {0}'.format(n))
            n_inner = n
            p = np.polyfit(x[idx], y[idx], 1)
            model = np.poly1d(p)
        else:
            continue
    return model

def run():
    x, y = genSamples()
    model = ransac(x, y, 10, 100)
    return x, y, model

if __name__ == '__main__':
    x, y, model = run()
    plt.plot(x, y , 'ro')
    plt.plot(x, model(x), 'bo')
    plt.show()
