import numpy as np

def sigmoid(t):
    """apply the sigmoid function on t."""
    # *****************
    # INSERT YOUR CODE HERE
    # TODO
    s=t
    ind_pos=np.where(t>=0)
    ind_neg=np.where(t<0)
    s[ind_pos]=1/(1+np.exp(-t[ind_pos]))
    s[ind_neg]=(np.exp(t[ind_neg]))/(1+np.exp(t[ind_neg]))
    return s

def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    # *****************
    # INSERT YOUR CODE HERE
    # TODO
    Xw=tx@w
    #print(sigmoid(Xw))
    return -(np.sum(y*np.log(sigmoid(Xw))+(np.ones(np.shape(y))-y)*np.log(np.ones(np.shape(y))-sigmoid(Xw))))
    
def calculate_loss1(y, tx, w):
    """compute the cost by negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)    
    

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    # *****************
    # INSERT YOUR CODE HERE
    # TODO
    d=y-sigmoid(tx@w)
    #print(d)
    return -tx.T@d#-np.sum(l.T,axis=1)


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    # *****************
    # INSERT YOUR CODE HERE
    # compute the loss: TODO
    loss=calculate_loss(y,tx,w)
    # *****************
    # *****************
    # INSERT YOUR CODE HERE
    # compute the gradient: TODO
    g=calculate_gradient(y,tx,w)
    print(np.linalg.norm(g))
    # *****************
    # *****************
    # INSERT YOUR CODE HERE
    # update w: TODO
    w=w-gamma*g
    # *****************
    return loss, w


def logistic_regression(y, tx, w):
    """return the loss, gradient, and Hessian."""
    # *****************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and Hessian: TODO
    loss=calculate_loss(y,tx,w)
    gradient=calculate_gradient(y,tx,w)
    Hessian=calculate_hessian(y,tx,w)
    return loss,gradient,Hessian


def logistic_regression_gradient_descent(y,tx,gamma,max_iter,threshold,w):
    # init parameters
    losses = []

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        print(loss)
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        #if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
         #   break
    return w,losses[iter]