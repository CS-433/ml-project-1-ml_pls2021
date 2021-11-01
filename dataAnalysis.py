import numpy as np
from standardize import*

def splitData(tX,label):
    ind = []
    
    for i in range(tX.shape[0]):
        if tX[i,22] in label:
            ind.append(i)
    return ind

def addIntercept(tX):
    return np.c_[np.ones((tX.shape[0],1)), tX]

def columnsMissingValue(tX,printing = 1):
    columns = []
    if printing == 0:
        print("Columns that contain meaningless values: ")
    for i in range(tX.shape[1]):
        count = 0
        proportion = 0
        for j in range(tX.shape[0]):
            if tX[j,i] == -999:
                count += 1
        if count != 0:
            proportion = count/tX.shape[0]
            if printing == 0:
                print("Column ",i, " contains meaningless values.", "Proportion: ", proportion)
        if proportion == 1.0:
            columns.append(i)
    return columns

def deleteColumns(tX,missColumns):
    tX = np.delete(tX,missColumns,1)
    return tX

def replaceMissingValue(tX,col,method):
    ind = np.where(tX[:,col]==-999)
    tX_col_clean = np.delete(tX[:,col],ind)
    if method == "delete":
        tX_del = np.delete(tX,ind,0)
        return tX_del
    elif method == "mean":
        mean = np.mean(tX_col_clean)
        tX[ind,col]= mean
        return tX
    elif method == "median":
        median = np.median(tX_col_clean)
        tX[ind,col]= median
        return tX

def recognize_high_corr(tX,threshold):
    C=np.corrcoef(tX.T)
    ind_=np.where(np.absolute(C)>=threshold)
    indices=[]
    for i in range(len(ind_[0])):
            if ind_[0][i]!=ind_[1][i]:
                indices.append((ind_[0][i],ind_[1][i]))
    return indices

def add_combination_col(tX,threshold):
    X_features = []
    indices=recognize_high_corr(tX,threshold)
    for i in range(tX.shape[1]):
        X_features.append(tX[:,i])
        for j in range(i+1):
            if not (i,j) in indices:
                X_features.append(np.multiply(tX[:,i],tX[:,j]))

    X_features = np.array(X_features)
    X_features = X_features.T
    return X_features

def add_combination_col_test(tX,indices):
    X_features = []
    for i in range(tX.shape[1]):
        X_features.append(tX[:,i])
        for j in range(i+1):
            if not (i,j) in indices:
                X_features.append(np.multiply(tX[:,i],tX[:,j]))

    X_features = np.array(X_features)
    X_features = X_features.T
    return X_features

def dataCleaning(tX,label,replacingMethod,correlation,threshold):

    if len(label) == 0:
        raise ValueError('No label given')

    ind = splitData(tX,label)
    tX_lab = tX[ind]

    # Searching for the columns that contain 100% of missing values. (0 or 1 to print them) 
    miss_lab = columnsMissingValue(tX_lab,printing = 1)
    
    # The label can be deleted.
    miss_lab.append(22)

    # The last column of label 0 only contains 0s so we delete this column.
    if 0 in label:
        miss_lab.append(tX_lab.shape[1]-1) 

    # Deleting the columns we don't need
    tX_lab = deleteColumns(tX_lab,miss_lab)

    # We decide what we do with the first column which contains some missing values (see porportions).
    # We can either replace them by the mean, the median or delete them completely.
    tX_lab = replaceMissingValue(tX_lab,0,replacingMethod)

    # We standardize the matrix
    tX_lab,mean,std = standardize(tX_lab)
    # Indices for the correlation for the test set
    indices=recognize_high_corr(tX_lab,threshold)
    #Linear combinations according to correlation matrix
    if correlation==True:
        tX_lab = add_combination_col(tX_lab,threshold)

    # Adding an intercept to each of the matrices
    tX_lab = addIntercept(tX_lab)

    return tX_lab, ind,indices,mean,std

def dataCleaningTest(tX,label,replacingMethod,mean,std,correlation,indices):
    
    if len(label) == 0:
        raise ValueError('No label given')

    ind = splitData(tX,label)
    tX_lab = tX[ind]

    # Searching for the columns that contain 100% of missing values. (0 or 1 to print them) 
    miss_lab = columnsMissingValue(tX_lab,printing = 1)
    
    # The label can be deleted.
    miss_lab.append(22)

    # The last column of label 0 only contains 0s so we delete this column.
    if 0 in label:
        miss_lab.append(tX_lab.shape[1]-1) 

    # Deleting the columns we don't need
    tX_lab = deleteColumns(tX_lab,miss_lab)

    # We decide what we do with the first column which contains some missing values (see porportions).
    # We can either replace them by the mean, the median or delete them completely.
    tX_lab = replaceMissingValue(tX_lab,0,replacingMethod)

    # We standardize the matrix
    tX_lab = standardize_test(tX_lab,mean,std)

    #Linear combinations according to correlation matrix
    if correlation==True:
        tX_lab = add_combination_col_test(tX_lab,indices)

    # Adding an intercept to each of the matrices
    tX_lab = addIntercept(tX_lab)
    
    return tX_lab,ind


