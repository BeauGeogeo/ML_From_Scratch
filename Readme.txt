*********************************************** Machine Learning Library from Scratch ***********************************************

The name is a bit ambitious, but here is the idea : programming some machine learning algorithms from scratch to better understand
their logic and some basic concepts of machine learning, such as normalization and scaling for example.

The code has been inspired by the scikit-learn library.


						*** Architecture of the project ***


There are 4 directories. 

The PreProcessing directory contains the script to deal with the preprocessing steps before training a mmachine learning model, like
methods to add the bias to the weight matrix, encoding categorical data or elaborating a grid search.

The PostProcessing directory contains the script to deal with the steps after training the model. For the moment, it is only to save
the model trained and its parameters.

The logistic and linear regression directories contain the file to perform logistic and linear regression respectively. More
information is to be found in the readme file of each directory. 