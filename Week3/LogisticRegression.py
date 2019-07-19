
# coding: utf-8

import numpy as np
import pandas as pd

# change 3 kinds of Iris to 1,2,3
# add a feature x0 = 1 (w * x + b to (b.w)*(x0,x))
def transform_classification(Iris_df):
    Iris = Iris_df.copy()
    classification = {}
    for i,c in enumerate(Iris.iloc[:,-1]):
        if c not in classification:
            classification[c] = len(classification) + 1
            Iris.iloc[i,-1] = len(classification)
        else:
            Iris.iloc[i,-1] = classification[Iris.iloc[i,-1]]
    Iris.insert(4,'10',pd.DataFrame(np.ones(len(Iris)+1)))
    return Iris

# def predict function
def predict(theta, x):
    a = [- (theta * x[i].T) for i in range(len(x))]
    h_theta = 1/(1 + np.exp(a))
    return h_theta

# define loss function
def eval_loss(theta, x, y):
    J_theta = -np.array([y[i] *np.log(predict(theta, x[i])) + (1 - y[i]) *np.log(1 - predict(theta, x[i])) for i in range(len(x))])
    clf = np.where(J_theta >= 0.5, 1, 0)
    correct_rate = (clf == y).mean()
    return J_theta.mean(), correct_rate

# calculate gradient
def get_gradient(y_pred, y_real, x):
    d_theta = np.array([((y_pred - y_real) *x[:,i]).mean() for i in range(x.shape[1])])
    return d_theta

# update theta
def update_theta(batch_x, batch_y, theta, lr):
    batch_y_pred = predict(theta, batch_x)
    d_theta = get_gradient(batch_y_pred, batch_y, batch_x)
    theta -= lr*d_theta
    return theta

# train function
def train(x, y, batch_size, epoch, lr):
    x = np.mat(x)
#     give theta a random value
    theta = np.random.random(x.shape[1])
    for epo in range(epoch):
#         chose samples randomly
        ids = np.random.choice(len(x), batch_size)
        batch_x = x[ids]
        batch_y = y[ids]
        theta = update_theta(batch_x, batch_y, theta, lr)
        loss, accuracy = eval_loss(theta, batch_x, batch_y)
        print('epoch:{}\nθ:{}\nloss = {}\naccuracy = {}\n'.format(epo+1, theta, loss, accuracy))
    return theta

def run():
    Iris = pd.read_csv('data/Iris.csv',index_col=0)
    Iris = transform_classification(Iris)
    Iris.iloc[:,-1] = np.where(Iris.values[:,-1] == 1, 1, 0)
#     split data to train 
    data = Iris.values
    idxs = np.array(list(range(len(data))))
    np.random.shuffle(idxs)
    k = int(1*len(idxs))
    train_x = data[idxs[:k],:-1]
    train_y = data[idxs[:k],-1]
    test_x =  data[idxs[k:],:-1]
    test_y =  data[idxs[k:],:1]
    
    lr = 0.001
    batch_size = 100
    epoch = 100
#     train
    theta = train(train_x[:,:], train_y, batch_size, epoch, lr)

if __name__ == '__main__':
    run()

