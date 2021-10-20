# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    e=y-tx.dot(w)
    print(e.shape)
    MSE=1/(2*len(y))*e.T.dot(e)
    return MSE

    # ***************************************************
    raise NotImplementedError