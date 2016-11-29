import numpy as np

def getMiniBatch(i, batch_size, train_set, one_hot):
    """
        Return a minibatch from the training set and the associated labels
        :param i: the identifier of the minibatch
        :param batch_size: the number of training examples
        :param train_set: the training set
        :param one_hot: the one-hot representation of the labels
        :type i: int
        :type batch_size: int
        :type train_set: ndarray
        :type ont_hot: ndarray
        :return: the minibatch of examples
        :return: the minibatch of labels
        :return: the number of examples in the minibatch
        :rtype: ndarray
        :rtype: ndarray
        :rtype: int
    """
    
    ### Mini-batch creation
    n_training = train_set[0].shape[0]
    idx_begin = i * batch_size
    idx_end = min((i+1) * batch_size, n_training)
    mini_batch_size = idx_end - idx_begin

    batch = train_set[0][idx_begin:idx_end]
    one_hot_batch = one_hot[:,idx_begin:idx_end]

    return np.asfortranarray(batch), one_hot_batch, mini_batch_size

def computeLoss(W, b, batch, labels, softmax):
    """
        Compute the loss value of the current network on the full batch
        :param W: the weights
        :param B: the bias
        :param batch: the weights
        :param labels: the bias
        :param act_func: the weights
        :type W: ndarray
        :type B: ndarray
        :type batch: ndarray
        :type act_func: function
        :return loss: the negative log-likelihood
        :return accuracy: the ratio of examples that are well-classified
        :rtype: float
        :rtype: float
    """
    
    ### Forward propagation
    H = W.dot(batch.T)+b

    ### Compute the softmax
    out = softmax(H)

    pred = np.argmax(out,axis=0)
    fy = out[labels,np.arange(np.size(labels))]
    try:
        loss = np.sum(-1. * np.log(fy))/np.size(labels)
    except Exception:
        fy[fy<1e-4] = fy[fy<1e-4] + 1e-6
        loss = np.sum(-1. * np.log(fy))/np.size(labels)
    accuracy = np.sum(np.equal(pred,labels))/float(np.size(labels))        
    
    return loss,accuracy

import matplotlib.pyplot as plt
from IPython import display
def update_line(hl, new_data):
    print(new_data)
    hl.set_xdata(np.append(hl.get_xdata(), new_data[0]))
    hl.set_ydata(np.append(hl.get_ydata(), new_data[1]))
    #display.clear_output(wait=True)
    #display.display(plt.gcf())
    plt.draw()
    plt.pause(0.01)