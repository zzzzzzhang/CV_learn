
# coding: utf-8

#recode LinearRegression in 'python's way'.use less for loop.
import numpy as np
import random
np.set_printoptions(suppress= True)
from matplotlib import pyplot as plt

#generate samples ;w:1,b:-2-2
def gen_sample(num):
    '''
    num: number of samples
    '''
    w = 3 + np.random.random()
    b = np.random.randint(-100,101,num)/5 + 100
    input_x = np.random.randint(0,101,num) * np.random.random(num)
    input_y = w * input_x + b
    print('w = {0} ,b = {1}'.format(w,b.mean()))
    return input_x, input_y

# inference, test, predict, same thing. Run model after training
def predict(w, b, x):
    return w*x + b

# evaluate loss
def eval_loss(w, b, x_real, y_real):
    '''
    use mean-squared error
    w: weight
    b: bias
    x_real: real x
    y_real: real y
    '''
    y_pred = predict(w, b, x_real)
    loss = 0.5*((y_pred - y_real)**2).mean()
    return loss

# get gradient
def get_gradient(y_pred, y_real, x):
    diff = y_pred - y_real
    dw = (diff * x).mean()
    db = diff.mean()
    return dw, db

# update gradient
def update_gradient(batch_x, batch_y , w, b, lr):
    batch_y_pred = predict(w, b, batch_x)
    dw,db = get_gradient(batch_y_pred, batch_y, batch_x)
    w -= dw*lr
    b -= db*lr
    return w, b

#train
def train(batch_size, x, y, epoch, lr):
    w = 0
    b = 0
#     get nums in each batch 
    for epo in range(epoch):
#         chose batch randomly
        batch_ids = np.random.choice(len(x),batch_size)
        batch_x = x[batch_ids]
        batch_y = y[batch_ids]
#         re-calculate w,b
        w, b = update_gradient(batch_x, batch_y, w, b, lr)
#         calculate loss
        loss = eval_loss(w, b, batch_x, batch_y)
    print('epoch:{},w:{},b:{}\nloss = {}\n'.format(epo+1, w, b, loss))
    return w, b
        
def run():
    input_x, input_y = gen_sample(10000)
    lr = 0.001
    batch_size = 5000
    epoch = 12000
    plt.plot(input_x,input_y,'b.')
    w, b = train(batch_size, input_x, input_y, epoch, lr)
    plt.plot(input_x,predict(w, b, input_x),'r-')

if __name__ == '__main__':
    run()

