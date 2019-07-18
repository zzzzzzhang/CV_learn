
# coding: utf-8

#recode LinearRegression in 'python's way'.use less for loop.
import numpy as np
import random
np.set_printoptions(suppress= True)

#generate samples ;w:10,b:10
def gen_sample(num):
    '''
    num: number of samples
    '''
    w = np.full(num,10) + np.random.rand(num)
    b = np.full(num,50) + (np.random.rand(num) * 20 - 10)
    input_x = np.arange(1,num+1)
    input_y = w * input_x + b
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
    loss = ((y_pred - y_real)**2).mean()
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

def run():
    input_x, input_y = gen_sample(1000)
    lr = 0.001
    batch_size = 100
    epoch = 5
    train(batch_size, input_x, input_y, epoch, lr)

if __name__ == '__main__':
    run()

