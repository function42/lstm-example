__author__ = 'rakesh'

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import lstm

def data8(n=100):
    # Generates '8' shaped data

    y = np.linspace(0,1, n)

    x = np.append(np.sin(2*np.pi*y), (-np.sin(2*np.pi*y)))

    return np.column_stack((x,np.append(y,y))).astype(dtype=np.float32)

data = data8()

lead_data = np.roll(data, 1)

model = lstm.LSTMCell(2, 2)

learning_rate = 1e-3
epoches = 100000

plt.scatter(data[:,0], data[:,1])
plt.show()

def train_with_sgd(model, x, y, learning_rate, epoches):
    '''
    Train model.
    '''

    for epoch in xrange(epoches):
        model.sgd_step(x, y, learning_rate)
        if(epoch % 100 == 0):
            [predictions,cell] = model.predict(x)

            plt.scatter(predictions[0], predictions[1] ,c='r')
            plt.show()
            print("Epoch: %s, Cost: %d",  (epoch, model.compute_cost(x, y)))

    return model

train_with_sgd(model, data, lead_data, learning_rate, epoches)