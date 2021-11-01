# Useful starting lines
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from dataAnalysis import*
from implementations import * 

#%load_ext autoreload
#%autoreload 2



from proj1_helpers import *
DATA_TRAIN_PATH = 'train.csv' # TODO: download train data and supply path here 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

tX_0, ind0,indices0,mean0,std0 = dataCleaning(tX,[0],"median",True,0.5) 
tX_1, ind1,indices1,mean1,std1 = dataCleaning(tX,[1],"median",True,0.5)
tX_2, ind2,indices2,mean2,std2 = dataCleaning(tX, [2], "median", True, 0.5)
tX_3, ind3,indices3,mean3,std3 = dataCleaning(tX, [3], "median", True,0.5)

y0=y[ind0]#putting the corresponding indices on the output variable
y1=y[ind1]
y2=y[ind2]
y3=y[ind3]

# Reg Logistic
#label0
gamma=8*10**-7
max_iter=10000
threshold = 1e-8
initial_w=np.zeros((tX_0.shape[1],))
#alpha_b=1
lambda_=10**-3
w0,loss=logistic_regression_gradient_descent_ridge(y0, tX_0,gamma,max_iter,threshold,initial_w,lambda_)
#w,loss=logistic_regression_gradient_descent_line_search(y0,tX_0,gamma,max_iter,threshold,initial_w,alpha_b)

#label1
gamma=8*10**-7
max_iter=15000
threshold = 1e-10
initial_w=-np.zeros((tX_1.shape[1],))
lambda_=10**-3
w1,loss1=logistic_regression_gradient_descent_ridge(y1, tX_1,gamma,max_iter,threshold,initial_w,lambda_)

#label2
gamma=2*10**-6
max_iter=15000
threshold = 1e-10
initial_w=-np.zeros((tX_2.shape[1],))
lambda_=10**-3
w2,loss2=logistic_regression_gradient_descent_ridge(y2, tX_2,gamma,max_iter,threshold,initial_w,lambda_)

#label3
gamma=5.5*10**-6
max_iter=15000
threshold = 1e-10
initial_w=-np.zeros((tX_3.shape[1],))
lambda_=10**-3
w3,loss3=logistic_regression_gradient_descent_ridge(y3, tX_3,gamma,max_iter,threshold,initial_w,lambda_)


# Prediction on the train
y_pred0 = predict_labels(w0,tX_0,"logistic")
y_pred1 = predict_labels(w1,tX_1,"logistic")
y_pred2 = predict_labels(w2,tX_2,"logistic")
y_pred3 = predict_labels(w3,tX_3,"logistic")

error0 = np.count_nonzero(y_pred0-y0)
error1= np.count_nonzero(y_pred1-y1)
error2 = np.count_nonzero(y_pred2-y2)
error3 = np.count_nonzero(y_pred3-y3)
print((error0+error1+error2+error3)/250000) #percentage of the mistakes on the train set

# Prediction on the test

DATA_TEST_PATH = 'test.csv' # TODO: download train data and supply path here 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

tX_te_0, ind0_te = dataCleaningTest(tX_test,[0],"median",mean0,std0,True,indices0) 
tX_te_1, ind1_te = dataCleaningTest(tX_test,[1],"median",mean1,std1,True,indices1)
tX_te_2, ind2_te = dataCleaningTest(tX_test, [2], "median",mean2,std2, True, indices2)
tX_te_3, ind3_te = dataCleaningTest(tX_test, [3], "median",mean3,std3, True, indices3)

OUTPUT_PATH = 'Logistic ridge cor' # TODO: fill in desired name of output file for submission
y_pred0 = predict_labels(w0, tX_te_0,"logistic")
y_pred1 = predict_labels(w1, tX_te_1,"logistic")
y_pred2 = predict_labels(w2, tX_te_2,"logistic")
y_pred3 = predict_labels(w3, tX_te_3,"logistic")
y_pred=np.zeros(len(ids_test))
y_pred[ind0_te]=y_pred0
y_pred[ind1_te]=y_pred1
y_pred[ind2_te]=y_pred2
y_pred[ind3_te]=y_pred3
y_pred[y_pred==0]=-1
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

