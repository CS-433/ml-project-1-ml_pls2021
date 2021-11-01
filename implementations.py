import numpy as np


###############least_squares_GD:

def compute_gradient(y, tx, w):
    e=y-tx.dot(w)
    grad=-1/(len(y))*tx.T.dot(e)
    return grad


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss=0
    for n_iter in range(max_iters):
        grad=compute_gradient(y,tx,w)
        loss=compute_loss(y,tx,w)
        w=w-gamma*grad
    return loss,w


def compute_loss(y, tx, w):
    e=y-tx.dot(w)
    MSE=1/(2*len(y))*e.T.dot(e)
    return MSE

###############least_squares_SGD:


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    err = y - tx@w
    grad = -tx.T@err / len(err)
    return grad, err


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
    w = initial_w
    loss=0
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            loss = compute_loss(y, tx, w)
    return loss, w

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
            
#######Least_square_and_ridge_regression:

    
def ridge_regression(y,tx, lambda_ ):
            lambda_I = lambda_ *np.identity(tx.shape[1])
            A = tx.T.dot(tx)+ lambda_I
            b = tx.T.dot(y)
            w = np.linalg.solve(A,b)
            loss = compute_loss(y,tx,w)
            return w,loss

##########Logistic:
def sigmoid(t):
#We change the implementation of the sigmoid function a little to avoid some non-desire values i.e inf or nan 
        ind_pos=np.where(t>=0)
        ind_neg=np.where(t<0)
        t[ind_pos]=1/(1+np.exp(-t[ind_pos]))
        t[ind_neg]=np.exp(t[ind_neg])/(1+np.exp(t[ind_neg]))
        return t

def calculate_loss(y, tx, w):
        Xw=tx@w
        return -np.sum(y*np.log(sigmoid(Xw))+(np.ones(np.shape(y))-y)*np.log(np.ones(np.shape(y))-sigmoid(Xw)))

def calculate_gradient(y, tx, w):
    d=y-sigmoid(tx@w)
    return -tx.T@d


def learning_by_gradient_descent(y, tx, w, gamma):
    loss=calculate_loss(y,tx,w)
    g=calculate_gradient(y,tx,w)
    w=w-gamma*g
    return loss, w


def logistic_regression(y, tx, w):
    loss=calculate_loss(y,tx,w)
    gradient=calculate_gradient(y,tx,w)
    Hessian=calculate_hessian(y,tx,w)
    return loss,gradient,Hessian


def logistic_regression_gradient_descent_line_search(y,tx,gamma,max_iter,threshold,w,alpha_b):
        # start the logistic regression
        losses=[]
        #Parameters for the line search
        c=0.7
        rho=1/2
        for iter in range(max_iter):
            # get loss and update w.
            alpha=alpha_b
            f=calculate_loss(y, tx, w)
            grad=calculate_gradient(y, tx, w)
            #Check that the iterations go inside the while loop
            valid=False
            while calculate_loss(y, tx, w+alpha*grad)<= f-c*alpha*grad.T@grad:
                alpha=alpha*rho
                valid=True
                print("yes")
            if valid==False:
                alpha=gamma
            loss, w = learning_by_gradient_descent(y, tx, w, alpha)
            # log info
            if iter % 100 == 0:
                print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            losses.append(loss)
            # converge criterion
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                 break
        return w,loss
def logistic_regression_gradient_descent(y,tx,gamma,max_iter,threshold,w):
        # start the logistic regression
        losses=[]
        for iter in range(max_iter):
            # get loss and update w.
            loss, w = learning_by_gradient_descent(y, tx, w, gamma)
            if iter % 100 == 0:
                print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            losses.append(loss)
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                   break
        return w,losses[iter]
    
    
######## Reg_logistic:

def calculate_loss_ridge(y,tx,w,lambda_):
    return calculate_loss(y,tx,w)+ lambda_*np.linalg.norm(w)**2
    
    
def calculate_gradient_ridge(y, tx, w, lambda_):
        d= sigmoid(tx@w) - y
        rhs = tx.T@d
        regularizer = 2 * lambda_ * w
        return rhs + regularizer


def learning_by_gradient_descent_ridge(y, tx, w, gamma, lambda_):
        loss=calculate_loss_ridge(y,tx,w,lambda_)
        g=calculate_gradient_ridge(y,tx,w,lambda_)
        w=w-gamma*g
        return loss, w, np.linalg.norm(g)


def logistic_regression_gradient_descent_ridge(y,tx,gamma,max_iter,threshold,w,lambda_):
        # init parameters
        losses = []

        # start the logistic regression
        for iter in range(max_iter):
            # get loss and update w.
            loss, w, grad= learning_by_gradient_descent_ridge(y, tx, w, gamma, lambda_)
            # log info
            if iter % 100 == 0:
                    print("Current iteration={i}, loss={l}, gradient norm = {g}".format(i=iter,l=loss, g = grad))
            # converge criterion
            losses.append(loss)
            #if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
             #   break

        return w,losses[iter]

