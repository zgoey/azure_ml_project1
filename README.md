# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset that we are investigating contains information about the success of direct marketing in a banking context. Based on various background characteristics, we seek to predict whether a client is susceptible to subscribe to a long term deposit (see also https://archive.ics.uci.edu/ml/datasets/bank+marketing and https://core.ac.uk/download/pdf/55631291.pdf).

The best performing model we have found for this problem using the aforementioned approaches is a logistic regression model that was trained  with a regularization strength of 4.4190 and a maximum of 100 iterations. This model attains an accuracy of 91.75%.

## Scikit-learn Pipeline
The Hyperdrive/Scikit-learn approach uses a logistic regression classifier, where hyperparameter tuning is applied to two hyperparameters:
1. regularization strength
2. maximum number of iterations

The dataset is first preprocessed by applying one-hot-encoding to the features "job", "contact" and "education" and label encoding to the other string-valued features. It is then divided in a train and test set using a 80-20 split. 

We use separate parameter samplers for each of the hyperparameters to be optimized:
1. *lognormal parameter sampler* for the regularization strength with a range between -3 and 3 <br/>
The benefit of this type of parameter sampler is that it enables us to uniformly sample the log of the regularization strength  instead of its sampling raw value.  This means that we can vary the order of magnitude of the regularization strength, which makes sense, since variations on a finer scale would probably not have much effect.
2. *choice sampler* for the maximum number of iterations sampling from {50, 100, 500, 1000, 5000, 10000}  <br/>
The benefit of using this sampler is that we can explicitly specify each of the values that we are interested in and only sample among those.

Using the aforementioned samplers and sampling ranges, we generate 100 hyperparameter combinations that are compared on their resulting classification accuracies on the test set.

To keep the computational burden in check, the hyperdrive run is configured with an early stopping policy called *bandit policy*, which uses an evaluation interval of 100 and a slack factor of 0.1. Runs that do not achieve at least 90% of the best accuracy found so far after 100 iterations are cancelled by this policy. Th benefit of this is a more efficient usage of resourced: the porceeing units are put to good use and are not wasting their time on computations that do not lead to a significant progress in achieving better accuracies.

The notebook udacity-project.ipynb shows the output of a hyperparameter run that results in a model with an accuracy of 91.75% using a regularization strength of 4.4190 and a maximum of 100 iterations.


## AutoML
The best model found by AutoML is a soft voting ensemble based on a XGBoostClassifier. The output of cell 10 of the notebook udacity-project.ipynb gives a detailed view of the hyperparameters used for this classifier. The voting ensemble attains an accuracy of 91.66%.

## Pipeline comparison
The logistic regression model used in the Scikit-learn pipeline is much simpler than the AutoML ensemble and also achieves a slightly better accuracy (91.75% vs 91.66%). The reason that AutoML does not find the logistic regression model is that it does not seem to be included in the list of models that were tried during the 30 minutes run. Besides that, the following other pipeline characteristics can cause differences:
- AutoML applies automatic featurization that is not included in the Scikit-learn pipeline, 
- AutoML does not necessarily the hyperparameter space in the way we specified for the Scikit-learn pipeline, 
- AutoML as used in our notebook assesses accuracy by applying a 5-fold cross validation as opposed to the simple train-test split applied in the Scikit-learn pipeline.

Of course, the slightly lower accuracy and increased model complexity is countered by the fact that AutoML really is fully automatic. It does not require the specification of a model architecture, and it also handles the choice of hyperparameters by itself.


## Future work
The maximum number of runs in the hyperparameter sampling step of the Scikit-learn pipeline can be increased to consider more hyperparameter combinations, which could improve the accuracy even further. To get a more stable accuracy estimate, 5-fold cross validation could also be applied in the Scikit-learn  pipeline, as it is in AutoML. The AutoML pipeline itself can be enhanced by enabling deep neural networks, which also could also lead to an increase in the achieved accuracy.

