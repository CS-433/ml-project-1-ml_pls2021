# ml-project-1-ml_pls2021
ml-project-1-ml_pls2021 created by GitHub Classroom
# General Information
The repository contains the code for Machine Learning course 2021 (CS-433) project 1.
## Team
The project is accomplished by team PLSteam with members: 

Leandre Castagna: @Defteggg \
Pascal  Epple   : @epplepascal \
Selima  Jaoua   : @salimajaoua


# Project structure
## Presentation : 
The data can be found on the GitHub of the course : https://github.com/ML_course/blob/master/projects/project1/data. Please to run our code, download the data and put it in the same folder of our files. \
proj1_helpers : we changed the function predict_label for two reasons : first, we modify the prediction from  -1,1 it became 0,1. Also, we added a input variable which tells us if the method is logistic or not, because depending of that, the prediction function will change. 
## Data analysis : 
dataAnalysis.py : process data for model by splitting the classes, delete missing values, and multiply features depending on the threshold. 
## Methods : 
implementations.py  : the implementation of 6 methods to train the model. 

cross_valisation.py : use cross-validation to find the best parameters for ridge regression. 

Folder Mains        : for each method, you can found a jupyter notebook file that output the prediction for both train and test data. To run this, please download this and place them in the same folder that the data and implementations.py
## Best model : 
run.py   : Results using the best model ( Regularized Logistic Regression ) for both train and test 
finalsubmission.csv : Prediction for the test data with our best model

## Report
report.pdf: a 2-pages report of the complete solution.
