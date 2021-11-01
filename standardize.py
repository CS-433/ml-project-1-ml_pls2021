import numpy as np

def standardize(x):

    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data,np.mean(x,axis=0),np.std(centered_data,axis=0)


#We standardize for the test according the mean and std of the train set
def standardize_test(x,mean,std):
    centered_data = x - mean
    std_data = centered_data / std
    
    return std_data