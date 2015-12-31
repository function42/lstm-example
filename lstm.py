__author__ = 'rakesh'

import numpy as np
import scipy as sp
from theano import tensor as T
import theano

class LSTMCell:

    def __init__(self, inputSize, hiddenSize):
        # Initialize LSTM cell and create cost functions and update rules.

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        concatSize = inputSize + hiddenSize

        '''
        Initialize network parameters.
        '''

        # Forget gate parameters.
        Wf = np.random.uniform(-np.sqrt(1/inputSize), np.sqrt(1/inputSize), (hiddenSize, concatSize))
        bf = np.zeros((hiddenSize), np.float32)

        # Input gate parameters.
        Wi = np.random.uniform(-np.sqrt(1/inputSize), np.sqrt(1/inputSize), (hiddenSize, concatSize))
        bi = np.zeros((hiddenSize), np.float32)

        # Cell gate parameters.
        Wc = np.random.uniform(-np.sqrt(1/inputSize), np.sqrt(1/inputSize), (hiddenSize, concatSize))
        bc = np.zeros((hiddenSize), np.float32)

        # Output function parameters.
        Wo = np.random.uniform(-np.sqrt(1/inputSize), np.sqrt(1/inputSize), (hiddenSize, concatSize))
        bo = np.zeros((hiddenSize), np.float32)

        # Make theano variables.
        self.Wf = theano.shared(Wf.astype(theano.config.floatX), 'Wf')
        self.bf = theano.shared(bf.astype(theano.config.floatX), 'bf')
        self.Wi = theano.shared(Wi.astype(theano.config.floatX), 'Wi')
        self.bi = theano.shared(bi.astype(theano.config.floatX), 'bi')
        self.Wc = theano.shared(Wc.astype(theano.config.floatX), 'Wc')
        self.bc = theano.shared(bc.astype(theano.config.floatX), 'bc')
        self.Wo = theano.shared(Wo.astype(theano.config.floatX), 'Wo')
        self.bo = theano.shared(bo.astype(theano.config.floatX), 'bo')

        # Initialize gradient variables.
        self.mWf = theano.shared(np.zeros(Wf.shape).astype(theano.config.floatX), 'dWf')
        self.mbf = theano.shared(np.zeros(bf.shape).astype(theano.config.floatX), 'dbf')
        self.mWi = theano.shared(np.zeros(Wi.shape).astype(theano.config.floatX), 'dWi')
        self.mbi = theano.shared(np.zeros(bi.shape).astype(theano.config.floatX), 'dbi')
        self.mWc = theano.shared(np.zeros(Wc.shape).astype(theano.config.floatX), 'dWc')
        self.mbc = theano.shared(np.zeros(bc.shape).astype(theano.config.floatX), 'dbc')
        self.mWo = theano.shared(np.zeros(Wo.shape).astype(theano.config.floatX), 'dWo')
        self.mbo = theano.shared(np.zeros(bo.shape).astype(theano.config.floatX), 'dbo')

        # Create theano graphs here.
        self.__buildModel__()

    def __buildModel__(self):
        Wf, bf, Wi, bi, Wc, bc, Wo, bo = self.Wf, self.bf, self.Wi, self.bi, self.Wc, self.bc, self.Wo, self.bo

        x = T.fmatrix('x')
        y = T.fmatrix('y')

        def forward_step(x_t, h_prev, c_prev):
            '''
            Compute hidden state in an LSTM
            :param x_t: Input vector
            :param h_prev: Hidden variable from previous time step.
            :param c_prev: Cell state from previous time step.
            :return: [new hidden variable, updated cell state]
            '''

            concat_vec = T.concatenate([x_t, h_prev])

            # Forget gate
            f_t = T.nnet.hard_sigmoid(Wf.dot(concat_vec) + bf)
            # Input gate
            i_t = T.nnet.hard_sigmoid(Wi.dot(concat_vec) + bi)

            # Cell update
            c_tilde_t = T.tanh(Wc.dot(concat_vec) + bc)
            c_t = f_t * c_prev + i_t * c_tilde_t

            # Hidden state
            O_t = T.nnet.hard_sigmoid(Wo.dot(concat_vec) + bo)
            h_t = O_t * T.tanh(c_t)

            return [h_t, c_t]

        [h_t, c_t], updates = theano.scan(
            forward_step,
            sequences=x,
            truncate_gradient=-1,
            outputs_info=[
                      dict(initial=T.zeros(self.hiddenSize)),
                      dict(initial=T.zeros(self.hiddenSize))])

        error = ((y - h_t) ** 2).sum()

        # Gradients
        dWf = T.grad(error, Wf)
        dWi = T.grad(error, Wi)
        dWc = T.grad(error, Wc)
        dWo = T.grad(error, Wo)

        dbf = T.grad(error, bf)
        dbi = T.grad(error, bi)
        dbc = T.grad(error, bc)
        dbo = T.grad(error, bo)

        self.predict = theano.function([x], [h_t, c_t])
        self.error = theano.function([x, y], error)
        self.bptt = theano.function([x, y], [dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo])

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # rmsprop cache updates
        mWf = decay * self.mWf + (1 - decay) * dWf ** 2
        mWi = decay * self.mWi + (1 - decay) * dWi ** 2
        mWc = decay * self.mWc + (1 - decay) * dWc ** 2
        mWo = decay * self.mWo + (1 - decay) * dWo ** 2
        mbf = decay * self.mbf + (1 - decay) * dbf ** 2
        mbi = decay * self.mbi + (1 - decay) * dbi ** 2
        mbc = decay * self.mbc + (1 - decay) * dbc ** 2
        mbo = decay * self.mbo + (1 - decay) * dbo ** 2

        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.Param(decay, default=0.9)],
            [],
            updates=[(Wf, Wf - learning_rate * dWf / T.sqrt(mWf + 1e-6)),
                     (Wi, Wi - learning_rate * dWi / T.sqrt(mWi + 1e-6)),
                     (Wc, Wc - learning_rate * dWc / T.sqrt(mWc + 1e-6)),
                     (Wo, Wo - learning_rate * dWo / T.sqrt(mWo + 1e-6)),
                     (bf, bf - learning_rate * dbf / T.sqrt(mbf + 1e-6)),
                     (bi, bi - learning_rate * dbi / T.sqrt(mbi + 1e-6)),
                     (bc, bc - learning_rate * dbc / T.sqrt(mbc + 1e-6)),
                     (bo, bo - learning_rate * dbo / T.sqrt(mbo + 1e-6)),
                     (self.mWf, mWf),
                     (self.mWi, mWi),
                     (self.mWc, mWc),
                     (self.mWo, mWo),
                     (self.mbf, mbf),
                     (self.mbi, mbi),
                     (self.mbc, mbc),
                     (self.mbo, mbo)
                    ]
        )

    def compute_cost(self, x, y):
        return self.error(x, y)
