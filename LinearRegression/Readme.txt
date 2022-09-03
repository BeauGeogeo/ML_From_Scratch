****************************************************************************************************** Linear Regression From Scratch ******************************************************************************************************


						*** Aim of the model ***


Here is the code for performing a linear regression from scratch.

We use it on the SOCR-HeightWeight dataset, trying to predict a person's weight according to its height. 

The interest of such a regression here is mainly theoretical. Even if predicting someone's weight according to its height could be
a way to check if the person's weight correspond to the standards, and how strong this correspondance is, we know that we can
observe significant differences in weight for a given height which are not necessseraly correlated to the person's health. 
Indeed, the relation between the height and the weight is pretty loose, as explained further.


						*** Important information about the dataset ***


Beware that despite the fact that the relation between height and weight has a general linear trend, the relation is pretty loose and
some notable differences can be observed for the same height. It is not the best example of a linear relation, even if again there is
a linear trend in the sense that the taller you are the heavier you might be. 


						*** Tasks performed by the script ***


We first plot the data to observe a posssible linear relation.

The data is normalized and scaled.

The model is then trained and predictions on the test set are made and plotted. 

Finally, the cost function is plotted.