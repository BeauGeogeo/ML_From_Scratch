***************************************************************************************************** Logistic Regression From Scratch *****************************************************************************************************


						*** Aim of the model ***


Here is the code for performing a logistic regression from scratch.

We use it on the Fractures dataset, a dataset containing examinations of people potentially injured and having fracture. The 
classification task consists in predicting wether or not a person has a fracture regarding characteristics such as age, medication, 
bone  mineral density (bmd), etc.

The interest of such a classification could be to prioritize or even rationalize access to radiographic examinations.


						*** Important information about the dataset ***


Beware that the dataset here is imbalanced. Although we got good results here, it could be a problem to deal with in other situations
and special techniques for managing imbalanced datasets could be required.


						*** Tasks performed by the script ***


We first do several 3D graphs to visualize possible linear relations between all possible sets of 2 variables taken from the features 
of the Fractures dataset and the target which is the presence or not of a fracture. These plots underline the importance of bmd, and 
it will be confirmed by the logistic regression.

We then stress the importance of normalizing and scaling the data. 

A numerical encoding of the categorical variables is also done. 

The model is then trained and predictions are made on a test set. Note the importance of choosing an appropriate threshold and the
consequences it can have on the model accuracy. Here the threshold is such that a prediction score >= 0.6 means that a fracture is
predicted. 

Finally, we perform a grid search and make predictions again with the best parameters found. The performances of the model are 
compared with those of a dummy classifier.