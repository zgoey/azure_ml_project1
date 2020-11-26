# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
The dataset that we are investigating contains information about the success of direct marketing in a banking context. Based on various background characteristics, we seek to predict whether a client is susceptible to subscribe to a long term deposit (see also https://archive.ics.uci.edu/ml/datasets/bank+marketing and https://core.ac.uk/download/pdf/55631291.pdf).

The best performing model we have found for this problem using the aforementioned Scitkit-learn and AutoML approaches is a logistic regression model with a regularization strength of 1.7341 and a maximum of 5000 iterations. This model attains an accuracy of 91.97%.

## Scikit-learn Pipeline
The Hyperdrive/Scikit-learn approach uses a logistic regression classifier, where hyperparameter tuning is applied to two hyperparameters:
1. regularization strength
2. maximum number of iterations 

The dataset is first preprocessed by applying one-hot-encoding to the features "job", "contact" and "education" and label encoding to the other string-valued features. It is then divided in a train and test set using a 80-20 split.

We use a lognormal parameter sampler for the regularization strength with a range between -5 and 3, which seems appropriate since we are more interested in varying the order of magnitude of this parameter as opposed to varying its magnitude itself.  For the maximum number of iterations, we use a choice sampler that picks values from the set {50, 100, 500, 1000, 5000, 10000}. The choice sampler has the advantage that we can explicitly specify each of the values that we are interested in.

We choose to apply a bandit policy with an evaluation interval of 100 and a slack factor of 0.1 to ensure that we do not unnecessarily use our computing resources without making a significant progress in achieving better accuracies.

Our notebook shows the output of a hyperparameter run that finds a model with an accuracy of 91.97% using a regularization strength of 1.7341 and a maximum of 5000 iterations.


## AutoML
The best model found by AutoML is a soft voting ensemble based on a XGBoostClassifier. The output of cell 10 of the notebook udacity-project.ipynb gives a detailed view of the hyperparameters used for this classifier. The voting ensemble attains an accuracy of 91.69%.

## Pipeline comparison
The logistic regression model used in the Scikit-learn pipeline is much simpler than the AutoML ensemble and also achieves a slightly better accuracy (91.97% vs 91.69%). The reason that AutoML does not find the logistic regression model is that it does not seem to be included in the list of models that were tried during the 30 minutes run. Besides that, the following other pipeline characteristics can cause differences:
- AutoML applies automatic featurization that is not included in the Scikit-learn pipeline, 
- AutoML does not necessarily the hyperparameter space in the way we specified for the Scikit-learn pipeline, 
- AutoML as used in our notebook assesses accuracy by applying a 5-fold cross validation as opposed to the simple train-test split applied in the Scikit-learn pipeline.

Of course, the slightly inferior accuracy and increased model complexity is countered by the fact that AutoML really is fully automatic. It does not require the specification of a model architecture and it also handles the choice of hyperparameters by itself.


## Future work
The maximum number of runs in the hyperparameter sampling step of the Scikit-learn pipeline can be increased to consider even more hyperparameter combinations, which could improve the accuracy even further. In addition to this, the AutoML experiment can be configured to enable deep neural networks, allowing more models to be considered, which also could benefit the achieved accuracy.

